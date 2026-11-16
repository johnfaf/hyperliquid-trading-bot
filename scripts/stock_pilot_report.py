#!/usr/bin/env python3
"""Write the previous US session's stock paper scorecard, without broker access."""
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.stock_evidence import MODELED_FEE_RATE, NEW_YORK, read_summary


def money(value):
    return "Unmeasured" if value is None else f"${value:,.2f}" if value >= 0 else f"-${-value:,.2f}"


def render(score):
    return "\n".join([
        f"# Stock Pilot: {score['session']}",
        "",
        f"**{score['state']}.** Simulated results only.",
        f"Strategy cycles: {score['cycles']}. Fresh quotes: {score['fresh_quotes']}/{score['quote_checks']} checked.",
        f"First / last observed marks (UTC): {score['first_mark'] or 'n/a'} / {score['last_mark'] or 'n/a'}.",
        "",
        f"| Change over observed window (USD per ${score['basis_usd']:,.0f}) | Result |",
        "| --- | ---: |",
        f"| Strategy | {money(score['strategy_move'])} |",
        f"| Cash | {money(score['cash_move'])} |",
        f"| Buy and hold (price only) | {money(score['hold_move'])} |",
        "",
        f"Strategy uses confirmed paper fills, marked inventory, and {MODELED_FEE_RATE * 10000:g} bps "
        "of modeled fees per fill. Broker fill prices already reflect the paper execution model. "
        "Supervised drill fills are excluded. No strategy performance is inferred from shadow orders.",
        "",
    ])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--day", type=date.fromisoformat)
    args = parser.parse_args()
    now = datetime.now(NEW_YORK)
    day = (args.day or (now.date() - timedelta(days=now.hour < 17))).isoformat()
    body = render(read_summary(args.db, session=day)["scorecard"])
    path = args.reports / f"stock-paper-{day}.md"
    with NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as output:
        temp = Path(output.name)
        output.write(body)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
