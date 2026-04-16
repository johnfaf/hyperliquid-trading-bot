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


def test_alpha_arena_backtester_uses_dynamic_risk_policy_across_future_bars(monkeypatch):
    from src.signals.alpha_arena import AlphaArena, ArenaAgent, AgentStatus
    from src.signals.signal_schema import RiskParams

    class _FakeRiskEngine:
        def apply(self, signal, regime_data=None, source_policy=None):
            signal.risk = RiskParams(
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                time_limit_hours=3.0,
                trailing_stop=False,
                risk_basis="price",
                enforce_reward_to_risk=False,
            )
            return signal

    monkeypatch.setattr(AlphaArena, "_init_db", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_load_agents", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_save_agents", lambda self: None)

    arena = AlphaArena(risk_policy_engine=_FakeRiskEngine())
    agent = ArenaAgent(
        agent_id="seed_momentum_long",
        name="Seed_momentum_long",
        strategy_type="momentum_long",
        status=AgentStatus.ACTIVE,
        params={
            "confidence_threshold": 0.5,
            "stop_loss_pct": 0.20,
            "take_profit_pct": 0.40,
            "position_pct": 0.05,
            "max_leverage": 2.0,
        },
    )
    signal = {"side": "long", "confidence": 0.8, "atr_pct": 0.02}
    entry_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000.0}
    future_bars = [
        {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 1_000.0},
        {"open": 100.8, "high": 105.5, "low": 100.6, "close": 104.8, "volume": 1_050.0},
    ]

    result = arena.backtester._simulate_trade(
        agent,
        signal,
        entry_bar,
        future_bars[0],
        [entry_bar],
        future_bars,
    )

    assert result is not None
    assert result["exit_price"] == pytest.approx(104.0)
    assert result["won"] is True


def test_alpha_arena_get_champion_signals_selects_best_coin(monkeypatch):
    from src.signals.alpha_arena import AlphaArena, ArenaAgent, AgentStatus

    def _bars(start: float, count: int = 40):
        return [
            {
                "open": start + i,
                "high": start + i + 0.8,
                "low": start + i - 0.6,
                "close": start + i + 0.4,
                "volume": 1_000.0 + i,
            }
            for i in range(count)
        ]

    monkeypatch.setattr(AlphaArena, "_init_db", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_load_agents", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_save_agents", lambda self: None)

    arena = AlphaArena()
    arena.agents = {}
    agent = ArenaAgent(
        agent_id="seed_momentum_long",
        name="Seed_momentum_long",
        strategy_type="momentum_long",
        status=AgentStatus.CHAMPION,
        params={"confidence_threshold": 0.5},
        total_trades=10,
        winning_trades=7,
        win_rate=0.7,
        sharpe_ratio=1.4,
        total_pnl=250.0,
        capital_allocated=1_000.0,
    )
    arena.agents[agent.agent_id] = agent

    def _fake_generate(agent_obj, bars, current_bar, coin="BTC"):
        if coin == "ETH":
            return {
                "coin": coin,
                "side": "long",
                "confidence": 0.82,
                "price": current_bar["close"],
                "atr_pct": 0.02,
            }
        if coin == "BTC":
            return {
                "coin": coin,
                "side": "long",
                "confidence": 0.58,
                "price": current_bar["close"],
                "atr_pct": 0.02,
            }
        return None

    monkeypatch.setattr(arena.backtester, "_agent_generate_signal", _fake_generate)

    signals = arena.get_champion_signals(
        current_candles={
            "BTC": _bars(100.0),
            "ETH": _bars(100.0),
        },
        min_fitness=0.15,
        min_trades=5,
        min_win_rate=0.45,
    )

    assert signals
    assert signals[0]["coin"] == "ETH"
    assert signals[0]["side"] == "long"


def test_alpha_arena_backtest_agent_aggregates_multi_coin_histories(monkeypatch):
    from src.signals.alpha_arena import AlphaArena, ArenaAgent, AgentStatus

    bars = [
        {
            "open": 100.0 + i,
            "high": 100.6 + i,
            "low": 99.4 + i,
            "close": 100.2 + i,
            "volume": 1_000.0 + i,
            "timestamp": i,
        }
        for i in range(70)
    ]

    monkeypatch.setattr(AlphaArena, "_init_db", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_load_agents", lambda self: None)
    monkeypatch.setattr(AlphaArena, "_save_agents", lambda self: None)

    arena = AlphaArena()
    agent = ArenaAgent(
        agent_id="seed_breakout",
        name="Seed_breakout",
        strategy_type="breakout",
        status=AgentStatus.ACTIVE,
        params={"confidence_threshold": 0.5},
    )

    seen_coins = set()

    def _fake_generate(agent_obj, visible, current_bar, coin="BTC"):
        seen_coins.add(coin)
        return {
            "coin": coin,
            "side": "long",
            "confidence": 0.7,
            "price": current_bar["close"],
            "atr_pct": 0.02,
        }

    def _fake_simulate(
        agent_obj,
        signal,
        entry_bar,
        exit_bar,
        history,
        future_bars,
        coin="BTC",
        sort_key=None,
    ):
        return {
            "coin": coin,
            "side": signal["side"],
            "entry_price": exit_bar["open"],
            "exit_price": exit_bar["close"],
            "pnl": 1.0,
            "return_pct": 0.01,
            "won": True,
            "_sort_key": sort_key or (0, coin),
        }

    monkeypatch.setattr(arena.backtester, "_agent_generate_signal", _fake_generate)
    monkeypatch.setattr(arena.backtester, "_simulate_trade", _fake_simulate)

    result = arena.backtester.backtest_agent(
        agent,
        {
            "BTC": bars,
            "ETH": bars,
        },
    )

    assert result["coins_tested"] == 2
    assert set(result["coin_results"]) == {"BTC", "ETH"}
    assert seen_coins == {"BTC", "ETH"}
