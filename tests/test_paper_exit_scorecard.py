from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import paper_exit_scorecard as scorecard
from src.backtest.paper_exit_study import ADVERSE_COSTS, POLICIES, Entry, ExitResult, Frame, evaluate_exit
from src.data.paper_exit_feed import MarketSample


ENTRY = Entry(17, 0, 100, 0.75, 98, 104)


def quote(at_ms, mark=100, *, bid_size=2):
    return MarketSample(
        at_ms, "ready", mark=mark, bid=mark - 0.02, ask=mark + 0.02, bid_size=bid_size,
        book_age_ms=0, context_age_ms=0, oracle_price=mark, funding_rate=0.0001,
    )


def test_adverse_replay_pends_same_trigger_through_rapid_reversal_and_thin_bid():
    frames = [Frame(s, None, None, None) for s in (
        quote(5_000, 104.5),  # TP triggers here in both cases.
        quote(10_000, 102, bid_size=0.1),  # Delay elapsed, but too little size.
        quote(15_000, 97),
    )]
    base = evaluate_exit(ENTRY, frames, 5_000, trailing=False)
    adverse = evaluate_exit(ENTRY, frames, 5_000, trailing=False, costs=ADVERSE_COSTS)
    assert base.trigger_ms == adverse.trigger_ms == 5_000
    assert base.reason == adverse.reason == "take_profit"
    assert base.exit_ms == 5_000
    assert adverse.exit_ms == 15_000
    assert adverse.exit_price == pytest.approx((97 - 0.04) * 0.999)
    assert base.modeled_pnl > 0 > adverse.modeled_pnl


def test_adverse_spread_and_fees_hurt_net_even_without_price_move():
    frames = [Frame(quote(5_000, 105), None, None, None), Frame(quote(10_000, 105), None, None, None)]
    base = evaluate_exit(ENTRY, frames, 5_000, trailing=False)
    adverse = evaluate_exit(ENTRY, frames, 5_000, trailing=False, costs=ADVERSE_COSTS)
    assert adverse.exit_price < base.exit_price
    assert adverse.modeled_pnl < base.modeled_pnl


def test_coverage_counts_invalid_mark_missing_sample_and_window_gap():
    samples = [quote(0), MarketSample(5_000, "disconnected"), quote(15_000),
               replace(quote(20_000), context_age_ms=60_000)]
    score = scorecard._market_coverage(samples, 0, 25_000)
    assert score.expected == 5
    assert score.samples == {"ready": 2, "disconnected": 1, "invalid": 1}
    assert score.max_ready_gap_ms == 15_000
    body = scorecard.render(score, 0, 25_000, 25_000)
    assert "no recorded sample: 1" in body
    assert "Not enough evidence" in body


def test_scorecard_excludes_cohort_closed_only_at_base_costs(monkeypatch):
    samples = [quote(i, 105 if i == 60_000 else 100) for i in range(0, 65_000, 5_000)]
    # All base arms trigger and fill at 60s; the adverse delay cannot be fabricated.
    monkeypatch.setattr(scorecard, "load_samples", lambda *_: samples)
    monkeypatch.setattr(scorecard, "load_entries", lambda *_: [ENTRY])
    score = scorecard.score_window(Path("ignored"), Path("ignored"), 0, 65_000, 65_000)
    assert score.entries == score.excluded == 1
    assert score.pairs == []
    assert score.reasons == {"at least one arm unclosed": 1}
    assert "No fully matched exits" in scorecard.render(score, 0, 65_000, 65_000)


def test_scorecard_reports_same_complete_cohort_in_both_cases(monkeypatch):
    samples = [quote(i, 105 if i >= 60_000 else 100) for i in range(0, 370_000, 5_000)]
    monkeypatch.setattr(scorecard, "load_samples", lambda *_: samples)
    monkeypatch.setattr(scorecard, "load_entries", lambda *_: [ENTRY])
    score = scorecard.score_window(Path("ignored"), Path("ignored"), 0, 370_000, 370_000)
    assert score.entries == len(score.pairs) == 1
    assert score.pairs[0].adverse["fixed_60s"].exit_ms == 65_000
    body = scorecard.render(score, 0, 370_000, 370_000)
    assert "paired in both scenarios: 1" in body
    assert "| `fixed_60s`" in body
    assert "Worst trade drawdown" in body


def _sufficient_score():
    result = scorecard.Score(expected=100, samples=Counter(ready=99), entries=30)
    opening = scorecard._ms(scorecard.HOLDOUT_START)
    fixed = ExitResult(exit_ms=1, modeled_pnl=1.0, profit_giveback=0.3, worst_trade_drawdown=0.6)
    trailing = replace(fixed, modeled_pnl=1.1, profit_giveback=0.2)
    for week in range(6):
        for k in range(5):
            at = opening + (7 * week + k % 3) * scorecard.DAY_MS + (k + 1) * 1_000
            policies = {name: trailing if name == "trailing_5s" else fixed for name in POLICIES}
            result.pairs.append(scorecard.Pair(at, "lower" if k % 2 else "elevated", policies, policies.copy()))
    return result


def test_gate_only_nominates_for_more_paper_and_rejects_adverse_degradation():
    score = _sufficient_score()
    assert scorecard.decision(score)[0] == "Candidate for further paper testing"
    adverse = score.pairs[0].adverse
    adverse["trailing_5s"] = replace(adverse["trailing_5s"], modeled_pnl=-20.0)
    verdict, reasons = scorecard.decision(score)
    assert verdict == "Keep fixed exits"
    assert any("adverse execution" in reason for reason in reasons)
    score.samples["ready"] = 98
    assert scorecard.decision(score)[0] == "Not enough evidence"


def test_gate_checks_advantage_survives_omitting_each_week():
    score = _sufficient_score()
    for i, pair in enumerate(score.pairs):
        revised = dict(pair.base)
        revised["trailing_5s"] = replace(revised["trailing_5s"], modeled_pnl=2.0 if i < 5 else 0.99)
        score.pairs[i] = replace(pair, base=revised)
    assert "week" in " ".join(scorecard.decision(score)[1])


def test_gate_rejects_entries_from_before_registration():
    score = _sufficient_score()
    score.pairs[0] = replace(score.pairs[0], at_ms=scorecard._ms(scorecard.HOLDOUT_START) - 1)
    with pytest.raises(ValueError, match="outside the holdout"):
        scorecard.decision(score)


def test_holdout_command_refuses_an_interim_verdict_before_opening_databases(monkeypatch, capsys):
    class FrozenDatetime:
        @staticmethod
        def now(_):
            return datetime(2026, 12, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(scorecard, "datetime", FrozenDatetime)
    with pytest.raises(SystemExit) as exc:
        scorecard.main(["--market-db", "ignored", "--pilot-db", "ignored", "--holdout"])
    assert exc.value.code == 2
    assert "no interim verdict" in capsys.readouterr().err
