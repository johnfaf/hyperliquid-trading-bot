"""
Structured Event Scanner
========================

Tracks official macro / central-bank events from public sources and normalizes
them into a single feed the dashboard can consume. This first version favors
official release calendars and release feeds over noisy headline scraping.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

import config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ET_TZ = ZoneInfo("America/New_York")

BLS_CALENDAR_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BLS_LATEST_RSS_URL = "https://www.bls.gov/feed/bls_latest.rss"
FED_MONETARY_RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
BEA_SCHEDULE_URL = "https://www.bea.gov/news/schedule"

DEFAULT_ASSETS = ["BTC", "ETH", "SOL"]
SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

_EVENT_RULES = [
    (("fomc", "federal open market", "interest rate", "policy decision", "statement from the federal open market committee"), "central_bank", "critical", True),
    (("consumer price index", "cpi"), "inflation", "critical", True),
    (("producer price index", "ppi"), "inflation", "high", True),
    (("employment situation", "nonfarm payroll", "payroll employment", "unemployment"), "labor", "critical", True),
    (("gross domestic product", "gdp"), "growth", "high", True),
    (("personal income and outlays", "personal income", "personal consumption expenditures", "pce"), "consumption", "high", True),
    (("retail sales",), "consumption", "high", True),
    (("job openings", "jolts"), "labor", "medium", True),
    (("international trade", "trade balance"), "trade", "medium", True),
    (("productivity", "unit labor costs"), "labor", "medium", True),
    (("advance economic indicators",), "growth", "medium", True),
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _parse_bool_like(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class EventScanner:
    """Fetch official macro / policy events and normalize them for the app."""

    def __init__(self, config_override: Optional[Dict] = None):
        cfg = dict(config_override or {})
        self.lookahead_days = int(cfg.get("lookahead_days", getattr(config, "EVENT_SCANNER_LOOKAHEAD_DAYS", 14)))
        self.recent_hours = int(cfg.get("recent_hours", getattr(config, "EVENT_SCANNER_RECENT_HOURS", 72)))
        self.refresh_seconds = int(cfg.get("refresh_seconds", getattr(config, "EVENT_SCANNER_REFRESH_SECONDS", 900)))
        self.max_upcoming = int(cfg.get("max_upcoming", getattr(config, "EVENT_SCANNER_MAX_UPCOMING", 12)))
        self.max_recent = int(cfg.get("max_recent", getattr(config, "EVENT_SCANNER_MAX_RECENT", 12)))
        self.include_medium = bool(cfg.get("include_medium", _parse_bool_like(getattr(config, "EVENT_SCANNER_INCLUDE_MEDIUM", True))))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "HyperliquidBotEventScanner/1.0 (+official-release-tracker)",
                "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9, */*;q=0.8",
            }
        )
        self._lock = threading.Lock()
        self._snapshot: Optional[Dict] = None
        self._last_refresh = 0.0
        self._source_status: Dict[str, Dict] = {}
        logger.info("EventScanner initialized")

    def _needs_refresh(self) -> bool:
        return (time.time() - self._last_refresh) >= max(self.refresh_seconds, 60)

    def _fetch_text(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("EventScanner fetch failed for %s: %s", url, exc)
            return None

    def _set_source_status(self, source: str, ok: bool, event_count: int = 0, error: str = "") -> None:
        self._source_status[source] = {
            "ok": ok,
            "event_count": event_count,
            "error": error[:160] if error else "",
            "updated_at": _utc_now().isoformat(),
        }

    def _classify_event(self, title: str, source: str) -> Dict:
        title_norm = _normalize_whitespace(title).lower()
        category = "macro"
        severity = "low"
        is_core = False
        tags: List[str] = []

        for keywords, rule_category, rule_severity, rule_core in _EVENT_RULES:
            if any(keyword in title_norm for keyword in keywords):
                category = rule_category
                severity = rule_severity
                is_core = rule_core
                tags.extend(keywords[:2])
                break

        if source == "Federal Reserve" and severity == "low":
            category = "central_bank"
            severity = "high"

        return {
            "category": category,
            "severity": severity,
            "severity_rank": SEVERITY_RANK[severity],
            "is_core": is_core,
            "assets": list(DEFAULT_ASSETS),
            "tags": tags[:3],
        }

    def _to_event(
        self,
        *,
        source: str,
        title: str,
        event_time: Optional[datetime],
        published_time: Optional[datetime],
        link: str = "",
        summary: str = "",
        source_type: str,
    ) -> Optional[Dict]:
        title = _normalize_whitespace(title)
        if not title:
            return None

        classification = self._classify_event(title, source)
        if classification["severity"] == "low" and not classification["is_core"]:
            return None
        if classification["severity"] == "medium" and not self.include_medium:
            return None

        now = _utc_now()
        reference_ts = event_time or published_time or now
        freshness_seconds = max(int((now - reference_ts).total_seconds()), 0) if reference_ts <= now else 0
        minutes_until = int((reference_ts - now).total_seconds() // 60) if reference_ts > now else 0
        status = "upcoming" if event_time and event_time > now else "recent"

        dedupe_key = f"{source}|{title.lower()}|{reference_ts.isoformat()}"
        return {
            "id": dedupe_key,
            "source": source,
            "source_type": source_type,
            "title": title,
            "summary": _normalize_whitespace(summary),
            "link": link,
            "category": classification["category"],
            "severity": classification["severity"],
            "severity_rank": classification["severity_rank"],
            "is_core": classification["is_core"],
            "assets": classification["assets"],
            "tags": classification["tags"],
            "event_time": event_time.isoformat() if event_time else None,
            "published_time": published_time.isoformat() if published_time else None,
            "status": status,
            "freshness_seconds": freshness_seconds,
            "minutes_until": minutes_until,
        }

    @staticmethod
    def _unfold_ics(text: str) -> List[str]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        unfolded: List[str] = []
        for line in lines:
            if not line:
                continue
            if line.startswith((" ", "\t")) and unfolded:
                unfolded[-1] += line[1:]
            else:
                unfolded.append(line)
        return unfolded

    def _parse_ics_datetime(self, raw_value: str, params: str) -> Optional[datetime]:
        raw_value = raw_value.strip()
        if not raw_value:
            return None
        try:
            if "VALUE=DATE" in params:
                dt = datetime.strptime(raw_value, "%Y%m%d")
                return dt.replace(tzinfo=ET_TZ).astimezone(timezone.utc)
            if raw_value.endswith("Z"):
                return datetime.strptime(raw_value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            dt = datetime.strptime(raw_value, "%Y%m%dT%H%M")
            tz_name_match = re.search(r"TZID=([^;:]+)", params)
            tz = ZoneInfo(tz_name_match.group(1)) if tz_name_match else ET_TZ
            return dt.replace(tzinfo=tz).astimezone(timezone.utc)
        except Exception:
            return None

    def _parse_bls_calendar(self, text: str) -> List[Dict]:
        now = _utc_now()
        cutoff = now + timedelta(days=self.lookahead_days)
        lines = self._unfold_ics(text)
        events: List[Dict] = []
        current: Dict[str, str] = {}

        def finalize_event() -> None:
            title = current.get("SUMMARY", "")
            dt = self._parse_ics_datetime(current.get("DTSTART", ""), current.get("DTSTART_PARAMS", ""))
            if not dt or dt < now or dt > cutoff:
                return
            event = self._to_event(
                source="BLS",
                title=title,
                event_time=dt,
                published_time=None,
                link="https://www.bls.gov/schedule/news_release/",
                summary=current.get("DESCRIPTION", ""),
                source_type="calendar",
            )
            if event:
                events.append(event)

        in_event = False
        for line in lines:
            if line == "BEGIN:VEVENT":
                in_event = True
                current = {}
                continue
            if line == "END:VEVENT":
                finalize_event()
                in_event = False
                current = {}
                continue
            if not in_event or ":" not in line:
                continue
            key, value = line.split(":", 1)
            if ";" in key:
                base_key, params = key.split(";", 1)
                current[f"{base_key}_PARAMS"] = params
                key = base_key
            current[key] = value
        return events

    @staticmethod
    def _html_to_lines(html: str) -> List[str]:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?i)<br\\s*/?>", "\n", text)
        text = re.sub(r"(?i)</p>|</div>|</li>|</tr>|</td>|</th>|</h\\d>", "\n", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = unescape(text)
        lines = [_normalize_whitespace(line) for line in text.splitlines()]
        return [line for line in lines if line]

    def _parse_bea_schedule(self, html: str) -> List[Dict]:
        now = _utc_now()
        cutoff = now + timedelta(days=self.lookahead_days)
        lines = self._html_to_lines(html)
        year_match = re.search(r"Year\s+(\d{4})\s+Release", html, flags=re.I)
        year = int(year_match.group(1)) if year_match else now.year
        date_pattern = re.compile(r"^[A-Z][a-z]+\s+\d{1,2}$")
        time_pattern = re.compile(r"^\d{1,2}:\d{2}\s+[AP]M$")
        skip_tokens = {"N", "News", "D", "Data", "V", "Visual Data", "Article"}

        events: List[Dict] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not date_pattern.match(line):
                i += 1
                continue
            if i + 2 >= len(lines) or not time_pattern.match(lines[i + 1]):
                i += 1
                continue

            date_str = f"{line} {year} {lines[i + 1]}"
            try:
                dt = datetime.strptime(date_str, "%B %d %Y %I:%M %p").replace(tzinfo=ET_TZ).astimezone(timezone.utc)
            except Exception:
                i += 1
                continue

            j = i + 2
            if lines[j].replace(" ", "") in {token.replace(" ", "") for token in skip_tokens}:
                j += 1
            if j >= len(lines):
                i += 1
                continue
            title = lines[j]
            if dt < now or dt > cutoff:
                i = j + 1
                continue

            event = self._to_event(
                source="BEA",
                title=title,
                event_time=dt,
                published_time=None,
                link=BEA_SCHEDULE_URL,
                summary="Official BEA release schedule",
                source_type="calendar",
            )
            if event:
                events.append(event)
            i = j + 1

        return events

    def _parse_rss_feed(self, text: str, source: str) -> List[Dict]:
        now = _utc_now()
        cutoff = now - timedelta(hours=self.recent_hours)
        events: List[Dict] = []
        try:
            root = ET.fromstring(text)
        except Exception as exc:
            logger.warning("EventScanner failed to parse %s feed: %s", source, exc)
            return events

        for item in root.findall(".//item"):
            title = item.findtext("title", default="") or ""
            link = item.findtext("link", default="") or ""
            description = item.findtext("description", default="") or ""
            raw_pub_date = item.findtext("pubDate", default="") or ""
            try:
                published = parsedate_to_datetime(raw_pub_date).astimezone(timezone.utc)
            except Exception:
                published = None
            if published and published < cutoff:
                continue
            event = self._to_event(
                source=source,
                title=title,
                event_time=None,
                published_time=published,
                link=link,
                summary=description,
                source_type="feed",
            )
            if event:
                events.append(event)
        return events

    @staticmethod
    def _dedupe(events: List[Dict]) -> List[Dict]:
        deduped: Dict[str, Dict] = {}
        for event in events:
            existing = deduped.get(event["id"])
            if existing is None or event["severity_rank"] > existing["severity_rank"]:
                deduped[event["id"]] = event
        return list(deduped.values())

    def scan_events(self, force: bool = False) -> Dict:
        with self._lock:
            if not force and self._snapshot and not self._needs_refresh():
                return self._snapshot

            upcoming: List[Dict] = []
            recent: List[Dict] = []

            ics_text = self._fetch_text(BLS_CALENDAR_ICS_URL)
            if ics_text:
                parsed = self._parse_bls_calendar(ics_text)
                upcoming.extend(parsed)
                self._set_source_status("BLS Calendar", True, len(parsed))
            else:
                self._set_source_status("BLS Calendar", False, 0, "fetch_failed")

            bea_html = self._fetch_text(BEA_SCHEDULE_URL)
            if bea_html:
                parsed = self._parse_bea_schedule(bea_html)
                upcoming.extend(parsed)
                self._set_source_status("BEA Schedule", True, len(parsed))
            else:
                self._set_source_status("BEA Schedule", False, 0, "fetch_failed")

            bls_rss = self._fetch_text(BLS_LATEST_RSS_URL)
            if bls_rss:
                parsed = self._parse_rss_feed(bls_rss, "BLS")
                recent.extend(parsed)
                self._set_source_status("BLS Feed", True, len(parsed))
            else:
                self._set_source_status("BLS Feed", False, 0, "fetch_failed")

            fed_rss = self._fetch_text(FED_MONETARY_RSS_URL)
            if fed_rss:
                parsed = self._parse_rss_feed(fed_rss, "Federal Reserve")
                recent.extend(parsed)
                self._set_source_status("Fed Monetary Feed", True, len(parsed))
            else:
                self._set_source_status("Fed Monetary Feed", False, 0, "fetch_failed")

            now = _utc_now()
            upcoming = self._dedupe(upcoming)
            recent = self._dedupe(recent)
            upcoming.sort(key=lambda item: (item["event_time"] or "", -item["severity_rank"], item["title"]))
            recent.sort(
                key=lambda item: (
                    item["published_time"] or item["event_time"] or "",
                    item["severity_rank"],
                ),
                reverse=True,
            )

            active = [
                event for event in upcoming
                if event["event_time"]
                and 0 <= event["minutes_until"] <= 24 * 60
                and event["severity_rank"] >= SEVERITY_RANK["high"]
            ]
            active.extend(
                event for event in recent
                if event["severity_rank"] >= SEVERITY_RANK["high"]
                and event["freshness_seconds"] <= 6 * 3600
            )
            active = self._dedupe(active)
            active.sort(key=lambda item: (-item["severity_rank"], item["minutes_until"], item["title"]))

            snapshot = {
                "timestamp": now.isoformat(),
                "upcoming": upcoming[: self.max_upcoming],
                "recent": recent[: self.max_recent],
                "active": active[:8],
                "summary": {
                    "upcoming_count": len(upcoming),
                    "recent_count": len(recent),
                    "active_count": len(active),
                    "high_impact_next_24h": sum(1 for event in active if event["status"] == "upcoming"),
                    "sources_ok": sum(1 for source in self._source_status.values() if source.get("ok")),
                    "sources_total": len(self._source_status),
                    "next_event_title": upcoming[0]["title"] if upcoming else "",
                    "next_event_time": upcoming[0]["event_time"] if upcoming else None,
                },
                "sources": dict(self._source_status),
            }
            self._snapshot = snapshot
            self._last_refresh = time.time()
            return snapshot

    def get_dashboard_data(self) -> Dict:
        """Return a dashboard-friendly event payload."""
        return self.scan_events(force=False)

    def get_stats(self) -> Dict:
        """Return compact stats for health/reporting."""
        snapshot = self.scan_events(force=False)
        return {
            "last_refresh_ts": snapshot.get("timestamp"),
            "upcoming_count": snapshot.get("summary", {}).get("upcoming_count", 0),
            "recent_count": snapshot.get("summary", {}).get("recent_count", 0),
            "active_count": snapshot.get("summary", {}).get("active_count", 0),
            "high_impact_next_24h": snapshot.get("summary", {}).get("high_impact_next_24h", 0),
            "sources": snapshot.get("sources", {}),
        }
