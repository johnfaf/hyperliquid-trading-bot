import sys
import types

import pytest


def test_llm_filter_applies_timeout_to_client_and_requests(monkeypatch):
    from src.signals.llm_filter import LLMFilter

    class _FakeResponse:
        content = [types.SimpleNamespace(text='{"decision":"approve","confidence_adjustment":1.0,"reason":"ok"}')]

    class _FakeClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.option_calls = []
            self.create_calls = []
            self.messages = types.SimpleNamespace(create=self._create)

        def with_options(self, **kwargs):
            self.option_calls.append(kwargs)
            return self

        def _create(self, **kwargs):
            self.create_calls.append(kwargs)
            return _FakeResponse()

    class _FakeAnthropicModule:
        __version__ = "test"
        last_client = None

        class Anthropic:
            def __new__(cls, **kwargs):
                client = _FakeClient(**kwargs)
                _FakeAnthropicModule.last_client = client
                return client

    monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropicModule)

    llm_filter = LLMFilter(
        {
            "anthropic_api_key": "test-key",
            "llm_timeout": 7,
            "llm_max_tokens": 64,
        }
    )
    approved, confidence, reason = llm_filter.filter(
        {"coin": "BTC", "side": "long", "confidence": 0.7, "features": {}},
        {"open_positions": [], "all_signals": []},
    )

    client = _FakeAnthropicModule.last_client
    assert approved is True
    assert confidence == pytest.approx(0.7)
    assert "approved" in reason
    assert client.init_kwargs["timeout"] == 7
    assert client.init_kwargs["max_retries"] == 0
    assert client.option_calls[0] == {"timeout": 7, "max_retries": 0}
    assert client.create_calls[0]["model"]


def test_paper_trader_prefers_rl_sizer(monkeypatch):
    from src.trading.paper_trader import PaperTrader

    class _Sizing:
        position_pct = 0.12
        position_usd = 1200.0
        has_edge = True
        win_rate = 0.6
        reward_risk_ratio = 1.8

    class _FakeRLSizer:
        def __init__(self):
            self.calls = []

        def get_sizing(self, **kwargs):
            self.calls.append(kwargs)
            return _Sizing()

    class _FakeKelly:
        def get_sizing(self, **kwargs):
            raise AssertionError("Kelly sizer should not be used when RL sizer is available")

    monkeypatch.setattr("src.trading.paper_trader.db.get_paper_account", lambda: {"balance": 10_000.0})

    rl_sizer = _FakeRLSizer()
    trader = PaperTrader(kelly_sizer=_FakeKelly(), rl_sizer=rl_sizer)

    sizing = trader._get_position_sizing(
        "momentum_long",
        10_000.0,
        0.8,
        signal={"coin": "BTC", "features": {"volatility": 0.031}},
        regime_data={"overall_regime": "trending_up"},
    )

    assert sizing.position_pct == pytest.approx(0.12)
    assert rl_sizer.calls[0]["regime"] == "trending_up"
    assert rl_sizer.calls[0]["recent_volatility"] == pytest.approx(0.031)


def test_copy_trader_prefers_rl_sizer(monkeypatch):
    from src.trading.copy_trader import CopyTrader

    class _Sizing:
        position_pct = 0.09
        position_usd = 900.0

    class _FakeRLSizer:
        def __init__(self):
            self.calls = []

        def get_sizing(self, **kwargs):
            self.calls.append(kwargs)
            return _Sizing()

    class _FakeKelly:
        def get_sizing(self, **kwargs):
            raise AssertionError("Kelly sizer should not be used when RL sizer is available")

    class _FakeForecaster:
        def predict_regime(self, coin):
            return {"regime": "crash"}

    rl_sizer = _FakeRLSizer()
    trader = CopyTrader(
        kelly_sizer=_FakeKelly(),
        rl_sizer=rl_sizer,
        regime_forecaster=_FakeForecaster(),
    )

    sizing = trader._get_position_sizing(
        {"coin": "ETH", "confidence": 0.7, "source_trader": "0xabc", "volatility": 0.025},
        10_000.0,
    )

    assert sizing.position_usd == pytest.approx(900.0)
    assert rl_sizer.calls[0]["strategy_key"] == "copy_trade:0xabc"
    assert rl_sizer.calls[0]["regime"] == "crash"
    assert rl_sizer.calls[0]["recent_volatility"] == pytest.approx(0.025)


def test_alpha_arena_uses_lstm_agent_for_lstm_direction(monkeypatch):
    from src.signals.alpha_arena import AlphaArena, ArenaAgent, AgentStatus

    class _FakeLSTM:
        def __init__(self):
            self.train_calls = 0
            self.generate_calls = 0

        def train(self, candles):
            self.train_calls += 1
            return {"samples": len(candles)}

        def generate_signal(self, candles):
            self.generate_calls += 1
            return {
                "side": "long",
                "confidence": 0.82,
                "price": candles[-1]["close"],
                "atr_pct": 0.021,
            }

    monkeypatch.setattr(AlphaArena, "_init_db", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_load_agents", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_save_agents", lambda self: None)

    lstm = _FakeLSTM()
    arena = AlphaArena(lstm_agent=lstm)
    bars = [
        {
            "open": 100.0 + (i * 0.4),
            "high": 101.0 + (i * 0.4),
            "low": 99.0 + (i * 0.4),
            "close": 100.5 + (i * 0.4) + (0.3 if i % 4 != 0 else -0.2),
            "volume": 1_000.0 + i,
        }
        for i in range(40)
    ]
    agent = ArenaAgent(
        agent_id="seed_lstm_direction",
        name="Seed_lstm_direction",
        strategy_type="lstm_direction",
        status=AgentStatus.ACTIVE,
        params={"confidence_threshold": 0.5},
    )

    signal = arena.backtester._agent_generate_signal(agent, bars, bars[-1])
    arena.run_cycle(historical_candles=bars)

    assert signal["side"] == "long"
    assert signal["confidence"] == pytest.approx(0.82)
    assert lstm.generate_calls == 1
    assert lstm.train_calls == 1
