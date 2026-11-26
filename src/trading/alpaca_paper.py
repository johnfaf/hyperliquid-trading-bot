"""Paper-only US equity broker transport. No production trading host is accepted."""
from decimal import Decimal
from datetime import datetime, timezone
from hashlib import sha256
import json
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.trading.order_journal import Fill

DEFAULT_SYMBOL = "SPY"
STRATEGY_ID = "equity_daily_sma20_v1"
REFERENCE_BALANCE_USD = 1000
MAX_ENTRY_USD = Decimal(75)
DAILY_LOSS_USD = Decimal(10)
MAX_QUOTE_AGE_SECONDS = 30
MAX_QUOTE_SPREAD_FRACTION = Decimal("0.005")


def paper_account_pin(account_id):
    if not isinstance(account_id, str) or not account_id or account_id != account_id.strip():
        raise ValueError("invalid paper account identity")
    return sha256(account_id.encode()).hexdigest()


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class AlpacaPaper:
    TRADING = "https://paper-api.alpaca.markets"
    DATA = "https://data.alpaca.markets"

    def __init__(self, key: str, secret: str):
        self.opener = build_opener(NoRedirects())
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "Content-Type": "application/json"}

    def request(self, path, *, query=None, payload=None, market_data=False):
        prefix = self.DATA if market_data else self.TRADING
        url = prefix + path + ("?" + urlencode(query) if query else "")
        request = Request(url, headers=self.headers, data=json.dumps(payload).encode() if payload is not None else None)
        with self.opener.open(request, timeout=8) as response:
            return json.load(response, parse_float=Decimal)

    def lookup(self, cid):
        try:
            return self.request("/v2/orders:by_client_order_id", query={"client_order_id": cid})
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def submit(self, intent):
        return self.request("/v2/orders", payload=intent)

    def fills(self) -> list[Fill]:
        token = None
        fills = []
        for _ in range(20):
            query = {"page_size": 100, "direction": "asc"}
            if token:
                query["page_token"] = token
            rows = self.request("/v2/account/activities/FILL", query=query)
            fills.extend(Fill(
                row["id"], row["order_id"], row["symbol"], row["side"],
                Decimal(row["qty"]), Decimal(row["price"]),
                datetime.fromisoformat(row["transaction_time"].replace("Z", "+00:00")).astimezone(timezone.utc).isoformat(),
            ) for row in rows)
            if len(rows) < 100:
                return fills
            if rows[-1]["id"] == token:
                break
            token = rows[-1]["id"]
        raise RuntimeError("broker fill history not fully available")
