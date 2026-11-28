from dataclasses import replace
from hashlib import sha256
import json

import pytest

from scripts import isolated_stock_exit_verification as verify
from scripts import stock_exit_test_alert as alerts
from scripts.isolated_stock_drill import PROTOCOL as LEGACY, Blocked, act
from tests.test_isolated_stock_drill import EXCLUDED, NOW, Paper


class ExitPaper(Paper):
    def __init__(self):
        super().__init__()
        self.split_sells = False
        self.market_open = True

    def submit(self, intent):
        row = super().submit(intent)
        row["filled_qty"] = intent["qty"]
        if self.split_sells and intent["side"] == "sell":
            fill = self.observed_fills.pop()
            qty = fill.quantity / 2
            self.observed_fills += [
                replace(fill, fill_id=fill.fill_id + "-1", quantity=qty),
                replace(
                    fill, fill_id=fill.fill_id + "-2", quantity=fill.quantity - qty
                ),
            ]
        return row

    def request(self, path, **kwargs):
        value = super().request(path, **kwargs)
        return {**value, "is_open": self.market_open} if path == "/v2/clock" else value


@pytest.fixture
def previous(tmp_path):
    broker = ExitPaper()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "legacy-db"
    for action in ("entry", "recover", "exit", "verify"):
        result = act(
            broker,
            source,
            action,
            expected=broker.identity,
            excluded=EXCLUDED,
            now=NOW,
            confirm=True,
        )
    (snapshot / "legacy-receipt").write_text(
        json.dumps({**result, "protocol": LEGACY, "action": "verify"})
    )
    return broker, snapshot, tmp_path / "exit.sqlite3"


def execute(previous, action, *, confirmed=True, exclude=EXCLUDED):
    broker, snapshot, target = previous
    return verify.execute(
        broker,
        target,
        action,
        expected=broker.identity,
        excluded=exclude,
        now=NOW,
        legacy_dir=snapshot,
        confirm=confirmed,
    )


def test_preflight_can_verify_account_while_market_closed_and_places_no_orders(
    previous,
):
    broker, snapshot, _ = previous
    before = (snapshot / "legacy-db").read_bytes()
    broker.market_open = False
    out = execute(previous, "preflight", confirmed=False)
    assert out["state"] == "preflight_verified_market_closed"
    assert out["flat"] and out["open_orders"] == 0 and not out["passed"]
    with pytest.raises(Blocked, match="window"):
        execute(previous, "entry")
    assert broker.post_count == 2
    assert (snapshot / "legacy-db").read_bytes() == before


@pytest.mark.parametrize("split", (False, True))
def test_lost_sell_ack_recovers_without_resubmit_and_reports_observed_fills(
    previous, split
):
    broker, snapshot, _ = previous
    before = (snapshot / "legacy-db").read_bytes()
    assert execute(previous, "entry")["state"] == "intent_sent"
    broker.split_sells = split
    assert execute(previous, "exit")["state"] == "submission_unknown"
    assert execute(previous, "exit")["state"] == "already_reserved_do_not_resubmit"
    # The same identity can be recovered again after a partial Activities outage.
    broker.delayed_fills = True
    assert (
        execute(previous, "verify", confirmed=False)["state"]
        == "broker_position_mismatch"
    )
    assert broker.post_count == 4
    broker.delayed_fills = False
    outcome = execute(previous, "verify", confirmed=False)
    assert outcome["passed"] and outcome["flat"] and outcome["new_orders"] == 2
    assert outcome["partial_sell_fill_observed"] is split
    assert broker.post_count == 4 and (snapshot / "legacy-db").read_bytes() == before


def test_unmanaged_history_excluded_account_and_unconfirmed_order_block(previous):
    broker, _, target = previous
    with pytest.raises(Blocked, match="Explicit"):
        execute(previous, "entry", confirmed=False)
    with pytest.raises(Blocked, match="cannot be used"):
        execute(
            previous,
            "preflight",
            exclude={sha256(broker.identity.encode()).hexdigest()},
        )
    assert not target.exists()
    broker.orders.append(
        {"id": "outside", "client_order_id": "outside", "status": "canceled"}
    )
    with pytest.raises(Blocked, match="Complete paper order history"):
        execute(previous, "entry")
    assert broker.post_count == 2


def test_telegram_alert_is_labeled_and_restarted_process_does_not_send_again(tmp_path):
    path = tmp_path / "delivery.db"
    posted = []

    def telegram(token, chat, body):
        posted.append(body)
        return 42

    assert (
        alerts.send_once(path, "123:mock-token", "456", send=telegram)["state"]
        == "accepted_by_telegram"
    )
    response = alerts.send_once(path, "123:mock-token", "456", send=telegram)
    assert response["duplicate_suppressed"] and not response["posted_this_run"]
    assert (
        len(posted) == 1
        and posted[0].startswith("[TEST ONLY]")
        and "0.07 SPY shares" in posted[0]
    )


def test_ambiguous_telegram_error_is_not_logged_or_retried(tmp_path):
    path = tmp_path / "delivery.db"
    assert alerts.send_once(path, "", "456")["state"] == "telegram_unconfigured"

    def timeout(*_):
        raise TimeoutError("secret URL")

    outcome = alerts.send_once(path, "123:mock-token", "456", send=timeout)
    assert outcome["state"] == "delivery_unknown" and "secret" not in str(outcome)
    assert alerts.send_once(path, "123:mock-token", "456", send=timeout)[
        "duplicate_suppressed"
    ]
