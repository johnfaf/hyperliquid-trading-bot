from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from scripts import paper_storage_capacity as storage
from scripts import publish_dashboard_status as publisher


def test_rollout_reserves_two_copies_including_wal(tmp_path, monkeypatch):
    db = tmp_path/"bot.db"
    wal = tmp_path/"bot.db-wal"
    orders = tmp_path/"bot.orders.sqlite3"
    stock = tmp_path/"stock.sqlite3"
    for path, size in ((db, 100), (wal, 40), (stock, 23), (orders, 90)):
        path.write_bytes(b"x"*size)
    rollback = tmp_path/"rollbacks"
    rollback.mkdir()
    monkeypatch.setattr(storage, "disk_usage", lambda p: SimpleNamespace(total=storage.GIB*40, free=storage.GIB*2))
    result = storage.measure(db,stock,rollback)
    assert result["paper_db_bytes"] == 100 and result["source_bytes"] == 253
    assert result["required"] == storage.RUN_RESERVE+2*253
    assert result["shortfall"] == 506


def test_capacity_monitor_uses_oldest_bounded_sample_and_sees_runway():
    now = datetime(2026,11,27,18,tzinfo=UTC)
    unit = storage.GIB
    sample = {"paper_db_bytes":14*unit,"active_free":12*unit,"active_total":80*unit}
    at = lambda hours: int((now-timedelta(hours=hours)).timestamp())
    history = [[at(80),unit],[at(-2),unit],[at(24),10*unit],[at(2),13*unit]]
    report, series = storage.track(sample, history, now)
    assert report["growth_bytes_per_day"] == 4*unit and report["days_to_reserve"] == 2.5
    assert report["severity"] == "low"
    assert series == [[at(24),10*unit],[at(2),13*unit],[at(0),14*unit]]
    steady, _ = storage.track({**sample, "paper_db_bytes":10*unit}, history, now)
    assert steady["days_to_reserve"] is None and steady["severity"] == "normal"


def test_storage_alerts_dedupe_per_severity_and_retry_failed_delivery(tmp_path, monkeypatch):
    now = datetime(2026,11,27,18,tzinfo=UTC)
    unit=storage.GIB
    snapshot={"severity":"low","active_free":8*unit,"active_total":80*unit,"shortfall":10*unit}
    state=tmp_path/"state.json"
    sent=[]

    def send(_, message):
        sent.append(message)
        return len(sent)>1

    monkeypatch.setattr(publisher,"send_telegram",send)
    assert publisher.notify_storage(snapshot,None,state,now) == "delivery_unavailable"
    assert not state.exists()
    assert publisher.notify_storage(snapshot,None,state,now) == "sent"
    assert publisher.notify_storage(snapshot,None,state,now+timedelta(hours=1)) == "quiet"
    assert publisher.notify_storage({**snapshot,"severity":"critical"},None,state,now) == "sent"
    assert all("Guarded rollout is short by 10.0 GiB" in m for m in sent)
