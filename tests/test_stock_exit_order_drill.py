from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from threading import Barrier

import pytest

from scripts import stock_exit_order_drill as drill
from scripts.run_stock_paper import submit_once


def test_recovery_restarts_processes_and_retries_only_the_confirmed_remainder(tmp_path):
    result = drill.exercise(tmp_path, "recovery")
    trace = {item["phase"]: item for item in result["trace"]}
    assert result["sell_posts"] == 2 and Decimal(result["remaining_qty"]) == 0
    assert trace["primary_ack_lost_after_post"]["state"] == "submission_unknown"
    assert trace["cancel_requested_not_confirmed"]["state"] == "blocked"
    assert (
        trace["canceled_but_late_fill_not_in_ledger"]["reason"]
        == "terminal_fill_evidence_pending"
    )
    assert (
        trace["broker_position_updated_before_fill_receipt"]["reason"]
        == "broker_position_mismatch"
    )
    assert (
        trace["replacement_process_exited_after_post"]["state"]
        == "worker_exited_before_ack"
    )
    assert Decimal(
        trace["replacement_process_exited_after_post"]["requested_qty"]
    ) == Decimal(".05")
    assert trace["replacement_lookup_missing_no_resubmit"]["state"] == "blocked"
    assert trace["restart_verified_flat"]["state"] == "flat_verified"
    venue = drill.FaultVenue(tmp_path / "venue.json", "recovery")
    assert [row[4] for row in venue.data["fills"] if row[3] == "sell"] == [
        "0.03",
        "0.02",
        "0.05",
    ]


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [("exit_unverified", ".10"), ("cancel_unconfirmed", ".07")],
)
def test_deadline_alert_is_durable_and_sells_are_blocked(tmp_path, scenario, expected):
    result = drill.exercise(tmp_path, scenario)
    alerts = [row for row in result["trace"] if row.get("new_alert")]
    assert len(alerts) == 1 and alerts[0]["alert"]["verified_flat"] is False
    assert result["sell_posts"] == 1 and Decimal(result["remaining_qty"]) == Decimal(
        expected
    )


def test_flat_is_not_declared_and_replacement_is_blocked_if_history_differs(tmp_path):
    scenario = "recovery"
    drill.FaultVenue.seed(tmp_path, scenario)
    assert drill.step(tmp_path, scenario)["state"] == "submission_unknown"
    venue = drill.FaultVenue(tmp_path / "venue.json", scenario)
    venue.change_primary("canceled")
    venue.data["orders"]["foreign"] = {
        "id": "foreign",
        "client_order_id": "foreign",
        "status": "canceled",
    }
    venue.save()
    assert drill.step(tmp_path, scenario)["reason"] == "terminal_fill_evidence_pending"
    assert len(drill.FaultVenue(venue.path, scenario).data["posts"]) == 1


def test_same_replacement_claim_is_atomic_across_journal_connections(tmp_path):
    the_intent = drill.intent("recovery", "replacement", Decimal(".05"))
    calls = []
    barrier = Barrier(2)

    class Broker:
        def submit(self, intent):
            calls.append(intent)
            return {
                "client_order_id": intent["client_order_id"],
                "id": "receipt",
                "status": "new",
            }

    def attempt():
        journal = drill.order_journal(tmp_path, "recovery")
        barrier.wait(timeout=5)
        return submit_once(Broker(), journal, the_intent, purpose=drill.PROTOCOL)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sorted(results) == ["intent_already_recorded_or_paused", "intent_sent"]
    assert calls == [the_intent]


def test_cli_produces_separate_non_approval_receipt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [str(drill.__file__), "--state-dir", str(tmp_path)])
    drill.main()
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["mode"] == "offline_fixture" and outcome["passed"]
    assert outcome["broker_order_submissions"] == 0
    assert not outcome["paper_approval"] and not outcome["live_approval"]
    assert outcome["source_sha256"] == drill.source_fingerprint()
    assert (tmp_path / outcome["run"] / "result.json").read_text() == (
        tmp_path / "latest.json"
    ).read_text()
