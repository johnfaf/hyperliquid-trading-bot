from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from src.data.event_scanner import (
    BEA_SCHEDULE_URL,
    BLS_CALENDAR_ICS_URL,
    BLS_LATEST_RSS_URL,
    ET_TZ,
    FED_MONETARY_RSS_URL,
    EventScanner,
)


def _future_et(days: int = 1, hour: int = 8, minute: int = 30) -> datetime:
    base = datetime.now(ET_TZ) + timedelta(days=days)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def test_parse_bls_calendar_extracts_core_releases_only():
    scanner = EventScanner({"lookahead_days": 30})
    cpi_dt = _future_et(days=2)
    misc_dt = _future_et(days=3, hour=13)
    ics = f"""BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;TZID=America/New_York:{cpi_dt.strftime('%Y%m%dT%H%M')}
SUMMARY:Consumer Price Index
DESCRIPTION:CPI release
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=America/New_York:{misc_dt.strftime('%Y%m%dT%H%M')}
SUMMARY:Regional office hours
DESCRIPTION:Not a core macro release
END:VEVENT
END:VCALENDAR
"""

    events = scanner._parse_bls_calendar(ics)

    assert len(events) == 1
    assert events[0]["source"] == "BLS"
    assert events[0]["category"] == "inflation"
    assert events[0]["severity"] == "critical"
    assert events[0]["status"] == "upcoming"


def test_parse_bea_schedule_extracts_release_rows():
    scanner = EventScanner({"lookahead_days": 30})
    bea_dt = _future_et(days=4)
    extra_dt = bea_dt + timedelta(days=1)
    html = f"""
    <html><body>
    <h1>Year {bea_dt.year} Release Schedule</h1>
    <div>{bea_dt.strftime('%B')} {bea_dt.day}</div>
    <div>{bea_dt.strftime('%I:%M %p').lstrip('0')}</div>
    <div>News</div>
    <div>Gross Domestic Product</div>
    <div>{extra_dt.strftime('%B')} {extra_dt.day}</div>
    <div>08:30 AM</div>
    <div>News</div>
    <div>Regional photography update</div>
    </body></html>
    """

    events = scanner._parse_bea_schedule(html)

    assert len(events) == 1
    assert events[0]["source"] == "BEA"
    assert events[0]["category"] == "growth"
    assert events[0]["title"] == "Gross Domestic Product"


def test_scan_events_combines_upcoming_and_recent_sources(monkeypatch):
    scanner = EventScanner(
        {
            "lookahead_days": 30,
            "recent_hours": 96,
            "max_upcoming": 10,
            "max_recent": 10,
        }
    )
    cpi_dt = _future_et(days=1)
    gdp_dt = _future_et(days=2)
    recent_pub = format_datetime(datetime.now(timezone.utc) - timedelta(hours=2))

    ics = f"""BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;TZID=America/New_York:{cpi_dt.strftime('%Y%m%dT%H%M')}
SUMMARY:Consumer Price Index
DESCRIPTION:CPI release
END:VEVENT
END:VCALENDAR
"""
    bea_html = f"""
    <html><body>
    <h1>Year {gdp_dt.year} Release Schedule</h1>
    <div>{gdp_dt.strftime('%B')} {gdp_dt.day}</div>
    <div>{gdp_dt.strftime('%I:%M %p').lstrip('0')}</div>
    <div>News</div>
    <div>Gross Domestic Product</div>
    </body></html>
    """
    bls_rss = f"""<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Producer Price Index</title>
        <link>https://www.bls.gov/example</link>
        <pubDate>{recent_pub}</pubDate>
        <description>PPI release</description>
      </item>
    </channel></rss>
    """
    fed_rss = f"""<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Statement from the Federal Open Market Committee</title>
        <link>https://www.federalreserve.gov/example</link>
        <pubDate>{recent_pub}</pubDate>
        <description>Monetary policy decision</description>
      </item>
    </channel></rss>
    """

    payloads = {
        BLS_CALENDAR_ICS_URL: ics,
        BEA_SCHEDULE_URL: bea_html,
        BLS_LATEST_RSS_URL: bls_rss,
        FED_MONETARY_RSS_URL: fed_rss,
    }
    monkeypatch.setattr(scanner, "_fetch_text", lambda url: payloads.get(url))

    snapshot = scanner.scan_events(force=True)

    assert snapshot["summary"]["sources_ok"] == 4
    assert snapshot["summary"]["upcoming_count"] == 2
    assert snapshot["summary"]["recent_count"] == 2
    assert snapshot["summary"]["high_impact_next_24h"] >= 1
    assert snapshot["upcoming"][0]["title"] == "Consumer Price Index"
    assert any(event["source"] == "Federal Reserve" for event in snapshot["recent"])


def test_scan_events_uses_cache_until_refresh(monkeypatch):
    scanner = EventScanner({"refresh_seconds": 3600})
    calls = {"n": 0}
    cpi_dt = _future_et(days=1)
    ics = f"""BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;TZID=America/New_York:{cpi_dt.strftime('%Y%m%dT%H%M')}
SUMMARY:Consumer Price Index
END:VEVENT
END:VCALENDAR
"""

    def _fake_fetch(url: str):
        calls["n"] += 1
        if url == BLS_CALENDAR_ICS_URL:
            return ics
        return None

    monkeypatch.setattr(scanner, "_fetch_text", _fake_fetch)

    first = scanner.scan_events(force=True)
    second = scanner.scan_events(force=False)

    assert first["summary"]["upcoming_count"] == 1
    assert second["summary"]["upcoming_count"] == 1
    assert calls["n"] == 4
