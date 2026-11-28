"""Deliver a single labeled Telegram fixture alert, with a durable send claim."""

import argparse
from contextlib import closing
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.request import HTTPRedirectHandler, Request, build_opener

EVENT = "STOCK-EXIT-TEST-01"
MESSAGE = (
    "[TEST ONLY] Stock exit drill (STOCK-EXIT-TEST-01)\n"
    "Simulated position: 0.07 SPY shares remain after an unconfirmed cancel.\n"
    "In a real incident, compare the original order ID, fills and remaining position in Alpaca "
    "Paper before doing anything. Do not submit another sell while the old order is unresolved.\n"
    "No broker order was placed by this alert. No action is required."
)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def deliver(token, chat, message):
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        method="POST",
        data=json.dumps({"chat_id": chat, "text": message}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with build_opener(NoRedirects()).open(request, timeout=8) as response:
        result = json.load(response)
    sent = result["result"] if result.get("ok") is True else {}
    target = sent.get("chat") or {}
    matches = (
        str(target.get("id")) == chat
        if not chat.startswith("@")
        else "@" + str(target.get("username", "")).lower() == chat.lower()
    )
    if (
        not matches
        or sent.get("text") != message
        or type(sent.get("message_id")) is not int
        or sent["message_id"] <= 0
    ):
        raise ValueError("Telegram acceptance not confirmed.")
    return sent["message_id"]


def credentials(path):
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}:
            if key in env:
                raise ValueError("Ambiguous Telegram configuration.")
            env[key] = value.strip().strip("'\"")
    return env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", "")


def send_once(path, token, chat, *, send=deliver):
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) or not re.fullmatch(
        r"-?[0-9]+|@[A-Za-z0-9_]+", chat
    ):
        return {"state": "telegram_unconfigured", "posted_this_run": False}
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("""CREATE TABLE IF NOT EXISTS deliveries (
            event TEXT PRIMARY KEY, state TEXT NOT NULL, message_id INTEGER, time TEXT NOT NULL
        )""")
        inserted = db.execute(
            "INSERT OR IGNORE INTO deliveries VALUES(?, 'reserved', NULL, ?)",
            (EVENT, datetime.now(UTC).isoformat()),
        ).rowcount
        state = db.execute(
            "SELECT state FROM deliveries WHERE event=?", (EVENT,)
        ).fetchone()[0]
    if not inserted:
        return {"state": state, "posted_this_run": False, "duplicate_suppressed": True}
    try:
        message_id = send(token, chat, MESSAGE)
        state = "accepted_by_telegram"
    except Exception:
        # Failed transport may conceal acceptance. Do not expose the URL or resend.
        state, message_id = "delivery_unknown", None
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute(
            "UPDATE deliveries SET state=?,message_id=? WHERE event=?",
            (state, message_id, EVENT),
        )
    return {"state": state, "posted_this_run": True, "duplicate_suppressed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--send-test", action="store_true", required=True)
    args = parser.parse_args()
    try:
        config = args.config or Path(os.environ["CREDENTIALS_DIRECTORY"]) / "telegram"
        result = send_once(args.state_dir / "delivery.sqlite3", *credentials(config))
    except (OSError, KeyError, ValueError, sqlite3.Error):
        result = {"state": "alert_state_unavailable_do_not_resend"}
    print(json.dumps({"event": EVENT, **result}))


if __name__ == "__main__":
    main()
