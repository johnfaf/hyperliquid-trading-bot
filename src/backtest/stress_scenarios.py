"""
Stress Scenario Library
========================
Realistic extreme-market event generators for the stress test engine.

Each scenario mutates a fill stream or injects synthetic fills that model
a specific market pathology.  All mutations are deterministic (seeded RNG)
so results reproduce across machines.

Scenarios modeled from actual Hyperliquid / crypto events:
  - Flash crash:  March 2020 COVID, May 2021 China ban, Nov 2022 FTX
  - Funding squeeze:  Hyperliquid OI squeeze events (shorts pay >100% ann.)
  - Liquidity drain:  DEX book thinning → slippage blowout
  - Cascade liquidation:  Leveraged long unwind → waterfall stops
  - Black swan (combined):  All of the above fire simultaneously
"""

import copy
import math
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


# ─── Scenario Configuration ───────────────────────────────────────

@dataclass
class FlashCrashConfig:
    """Models a sudden vertical price drop + recovery."""
    enabled: bool = True
    name: str = "Flash Crash"
    # Crash dynamics
    drop_pct: float = 0.25           # 25% price drop at trough
    duration_minutes: int = 12       # total event duration
    recovery_pct: float = 0.60       # recovers 60% of the drop (dead-cat bounce)
    # Cascade: each open long loses more due to slippage during panic
    panic_slippage_mult: float = 8.0 # slippage 8x normal during crash
    # How many fills get injected into the crash window
    n_crash_fills: int = 40
    # When in the fill stream to inject (0.0-1.0 = fraction of timeline)
    inject_at_pct: float = 0.45
    seed: int = 100


@dataclass
class FundingSqueezeConfig:
    """Models extreme funding rates that bleed leveraged positions."""
    enabled: bool = True
    name: str = "Funding Rate Squeeze"
    # Funding rate dynamics (per 8h)
    peak_funding_rate_8h: float = 0.015   # 1.5% per 8h = 547% annualized
    squeeze_duration_hours: int = 72      # 3-day squeeze
    ramp_up_hours: int = 12               # funding ramps over 12h
    # Which side gets squeezed
    squeeze_side: str = "long"            # longs pay shorts
    # Price impact: during funding squeeze, price drifts against squeezed side
    price_drift_pct: float = -0.08        # -8% drift over squeeze duration
    inject_at_pct: float = 0.55
    seed: int = 200


@dataclass
class LiquidityDrainConfig:
    """Models orderbook thinning → massive slippage on every trade."""
    enabled: bool = True
    name: str = "Liquidity Evaporation"
    # Slippage dynamics
    normal_slippage_bps: float = 4.5      # normal: 4.5 bps
    crisis_slippage_bps: float = 150.0    # crisis: 150 bps (1.5%)
    drain_duration_hours: int = 48        # 2 days of thin books
    ramp_up_hours: int = 6                # books thin over 6h
    # Partial fill rate increases (can't fill full size)
    partial_fill_rate: float = 0.60       # 60% of orders only partially fill
    partial_fill_min_pct: float = 0.20    # min 20% filled
    inject_at_pct: float = 0.35
    seed: int = 300


@dataclass
class CascadeLiquidationConfig:
    """Models a leveraged-long unwind that triggers a liquidation waterfall."""
    enabled: bool = True
    name: str = "Cascade Liquidation"
    # Liquidation dynamics
    initial_drop_pct: float = 0.08        # 8% triggers first wave
    cascade_waves: int = 4                # 4 waves of liquidations
    wave_amplification: float = 1.4       # each wave is 1.4x the previous
    wave_interval_minutes: int = 8        # 8 min between waves
    # Total event: ~32 min, total drop ~25-35%
    recovery_pct: float = 0.30            # weak bounce (deleveraging continues)
    n_fills_per_wave: int = 15
    inject_at_pct: float = 0.65
    seed: int = 400


@dataclass
class BlackSwanConfig:
    """Combined scenario: all pathologies fire within a 6-hour window."""
    enabled: bool = True
    name: str = "Black Swan (Combined)"
    # Timing: everything happens in a 6h window
    window_hours: int = 6
    # Component severity (multipliers on individual scenario defaults)
    crash_severity: float = 1.5           # 50% worse than standalone flash crash
    funding_severity: float = 2.0         # 2x peak funding
    liquidity_severity: float = 1.8       # 80% worse slippage
    liquidation_severity: float = 1.3     # 30% more cascade waves
    inject_at_pct: float = 0.75
    seed: int = 500


@dataclass
class StressScenarioSuite:
    """Full suite of scenarios to run."""
    flash_crash: FlashCrashConfig = field(default_factory=FlashCrashConfig)
    funding_squeeze: FundingSqueezeConfig = field(default_factory=FundingSqueezeConfig)
    liquidity_drain: LiquidityDrainConfig = field(default_factory=LiquidityDrainConfig)
    cascade_liquidation: CascadeLiquidationConfig = field(default_factory=CascadeLiquidationConfig)
    black_swan: BlackSwanConfig = field(default_factory=BlackSwanConfig)

    def enabled_scenarios(self) -> list:
        """Return list of (name, config) for all enabled scenarios."""
        out = []
        for attr in ["flash_crash", "funding_squeeze", "liquidity_drain",
                      "cascade_liquidation", "black_swan"]:
            cfg = getattr(self, attr)
            if cfg.enabled:
                out.append((cfg.name, attr, cfg))
        return out

    def to_dict(self) -> Dict:
        return {
            "flash_crash": asdict(self.flash_crash),
            "funding_squeeze": asdict(self.funding_squeeze),
            "liquidity_drain": asdict(self.liquidity_drain),
            "cascade_liquidation": asdict(self.cascade_liquidation),
            "black_swan": asdict(self.black_swan),
        }


# ─── Scenario Injectors ──────────────────────────────────────────
# Each injector takes a fill list and returns a mutated copy.
# Fills are dicts with: wallet_address, coin, side, original_price,
# penalised_price, size, time_ms, delayed_time_ms, closed_pnl,
# penalised_pnl, fee, is_liquidation, direction.

def inject_flash_crash(fills: list, cfg: FlashCrashConfig,
                        coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """
    Inject a flash crash into the fill stream.

    Mutates prices in a window around inject_at_pct, creating a V-shaped
    price drop with partial recovery.  Also injects synthetic panic-sell
    fills at the worst prices.
    """
    rng = random.Random(cfg.seed)
    fills = [copy.deepcopy(f) for f in fills]
    if not fills:
        return fills, {}

    coins = coins or list({f["coin"] for f in fills})
    crash_coin = rng.choice(coins)

    # Find injection point
    t_min = fills[0]["time_ms"]
    t_max = fills[-1]["time_ms"]
    t_start = int(t_min + (t_max - t_min) * cfg.inject_at_pct)
    t_duration = cfg.duration_minutes * 60 * 1000
    t_trough = t_start + t_duration // 3           # trough at 1/3
    t_end = t_start + t_duration

    # Get reference price at injection point
    ref_price = None
    for f in fills:
        if f["coin"] == crash_coin and f["time_ms"] <= t_start:
            ref_price = f["original_price"]
    if ref_price is None:
        ref_price = next((f["original_price"] for f in fills
                          if f["coin"] == crash_coin), 1000.0)

    trough_price = ref_price * (1 - cfg.drop_pct)
    recovery_price = trough_price + (ref_price - trough_price) * cfg.recovery_pct

    # Mutate existing fills in the crash window
    mutated = 0
    for f in fills:
        if f["coin"] != crash_coin:
            continue
        t = f["time_ms"]
        if t < t_start or t > t_end:
            continue

        # Compute price at this point in the crash curve
        if t <= t_trough:
            # Drop phase: linear to trough
            progress = (t - t_start) / max(1, t_trough - t_start)
            crash_price = ref_price - (ref_price - trough_price) * progress
        else:
            # Recovery phase: linear bounce
            progress = (t - t_trough) / max(1, t_end - t_trough)
            crash_price = trough_price + (recovery_price - trough_price) * progress

        # Apply panic slippage
        slippage = cfg.panic_slippage_mult * 4.5 / 10_000
        f["original_price"] = round(crash_price, 6)
        if f["side"] == "buy":
            f["penalised_price"] = round(crash_price * (1 + slippage), 6)
        else:
            f["penalised_price"] = round(crash_price * (1 - slippage), 6)
        mutated += 1

    # Inject synthetic panic-sell fills at the worst prices
    injected = []
    for i in range(cfg.n_crash_fills):
        t = t_start + int(rng.uniform(0, t_duration * 0.4))  # front-loaded
        progress = min(1.0, (t - t_start) / max(1, t_trough - t_start))
        price = ref_price - (ref_price - trough_price) * progress
        slippage = cfg.panic_slippage_mult * 4.5 / 10_000

        fill = {
            "wallet_address": "0xstress_flash_crash",
            "coin": crash_coin,
            "side": "sell",
            "original_price": round(price, 6),
            "penalised_price": round(price * (1 - slippage), 6),
            "size": round(rng.uniform(0.01, 0.5), 6),
            "time_ms": t,
            "delayed_time_ms": t + 2000,
            "closed_pnl": 0.0,
            "penalised_pnl": 0.0,
            "fee": round(abs(price * 0.001), 4),
            "is_liquidation": 1 if rng.random() < 0.3 else 0,
            "direction": "Close Long",
        }
        injected.append(fill)

    fills.extend(injected)
    fills.sort(key=lambda f: f["time_ms"])

    meta = {
        "coin": crash_coin,
        "ref_price": ref_price,
        "trough_price": round(trough_price, 2),
        "recovery_price": round(recovery_price, 2),
        "drop_pct": round(cfg.drop_pct * 100, 1),
        "fills_mutated": mutated,
        "fills_injected": len(injected),
        "t_start_ms": t_start,
        "t_end_ms": t_end,
    }
    return fills, meta


def inject_funding_squeeze(fills: list, cfg: FundingSqueezeConfig,
                            coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """
    Simulate a funding rate squeeze by mutating fills to include
    funding costs and adverse price drift.
    """
    rng = random.Random(cfg.seed)
    fills = [copy.deepcopy(f) for f in fills]
    if not fills:
        return fills, {}

    coins = coins or list({f["coin"] for f in fills})
    target_coin = rng.choice(coins)

    t_min = fills[0]["time_ms"]
    t_max = fills[-1]["time_ms"]
    t_start = int(t_min + (t_max - t_min) * cfg.inject_at_pct)
    t_ramp_end = t_start + cfg.ramp_up_hours * 3600 * 1000
    t_end = t_start + cfg.squeeze_duration_hours * 3600 * 1000

    # Get reference price
    ref_price = None
    for f in fills:
        if f["coin"] == target_coin and f["time_ms"] <= t_start:
            ref_price = f["original_price"]
    if ref_price is None:
        ref_price = next((f["original_price"] for f in fills
                          if f["coin"] == target_coin), 1000.0)

    mutated = 0
    total_funding_cost = 0.0

    for f in fills:
        if f["coin"] != target_coin:
            continue
        t = f["time_ms"]
        if t < t_start or t > t_end:
            continue

        # Funding rate curve: ramps up then holds
        if t <= t_ramp_end:
            progress = (t - t_start) / max(1, t_ramp_end - t_start)
            funding_rate = cfg.peak_funding_rate_8h * progress
        else:
            # Plateau then slow decay
            decay_progress = (t - t_ramp_end) / max(1, t_end - t_ramp_end)
            funding_rate = cfg.peak_funding_rate_8h * (1 - decay_progress * 0.3)

        # Price drift against squeezed side
        drift_progress = (t - t_start) / max(1, t_end - t_start)
        drift = ref_price * cfg.price_drift_pct * drift_progress
        new_price = ref_price + drift

        f["original_price"] = round(new_price, 6)
        # Penalised price includes both slippage AND funding cost
        funding_penalty = abs(new_price * funding_rate * 0.1)  # per-fill funding est
        total_funding_cost += funding_penalty
        if f["side"] == "buy":
            f["penalised_price"] = round(new_price * 1.00045 + funding_penalty, 6)
        else:
            f["penalised_price"] = round(new_price * 0.99955 - funding_penalty, 6)
        f["fee"] = round(f.get("fee", 0) + funding_penalty, 4)
        mutated += 1

    meta = {
        "coin": target_coin,
        "peak_funding_8h": f"{cfg.peak_funding_rate_8h:.2%}",
        "annualized": f"{cfg.peak_funding_rate_8h * 3 * 365:.0%}",
        "squeeze_hours": cfg.squeeze_duration_hours,
        "price_drift": f"{cfg.price_drift_pct:.1%}",
        "fills_mutated": mutated,
        "total_funding_cost": round(total_funding_cost, 2),
    }
    return fills, meta


def inject_liquidity_drain(fills: list, cfg: LiquidityDrainConfig,
                            coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """
    Simulate orderbook thinning: massive slippage + partial fills.
    """
    rng = random.Random(cfg.seed)
    fills = [copy.deepcopy(f) for f in fills]
    if not fills:
        return fills, {}

    t_min = fills[0]["time_ms"]
    t_max = fills[-1]["time_ms"]
    t_start = int(t_min + (t_max - t_min) * cfg.inject_at_pct)
    t_ramp_end = t_start + cfg.ramp_up_hours * 3600 * 1000
    t_end = t_start + cfg.drain_duration_hours * 3600 * 1000

    mutated = 0
    partial_fills = 0

    for f in fills:
        t = f["time_ms"]
        if t < t_start or t > t_end:
            continue

        # Slippage ramp: normal → crisis over ramp_up_hours
        if t <= t_ramp_end:
            progress = (t - t_start) / max(1, t_ramp_end - t_start)
        else:
            # Plateau then slow recovery
            decay = (t - t_ramp_end) / max(1, t_end - t_ramp_end)
            progress = max(0, 1.0 - decay * 0.5)

        current_slippage = (cfg.normal_slippage_bps +
                            (cfg.crisis_slippage_bps - cfg.normal_slippage_bps) * progress)
        slippage_frac = current_slippage / 10_000

        price = f["original_price"]
        if f["side"] == "buy":
            f["penalised_price"] = round(price * (1 + slippage_frac), 6)
        else:
            f["penalised_price"] = round(price * (1 - slippage_frac), 6)

        # Partial fill simulation
        if rng.random() < cfg.partial_fill_rate * progress:
            fill_pct = rng.uniform(cfg.partial_fill_min_pct, 0.7)
            f["size"] = round(f["size"] * fill_pct, 6)
            partial_fills += 1

        f["fee"] = round(abs(price * slippage_frac * 2 * f["size"]), 4)
        mutated += 1

    meta = {
        "peak_slippage_bps": cfg.crisis_slippage_bps,
        "drain_hours": cfg.drain_duration_hours,
        "fills_mutated": mutated,
        "partial_fills": partial_fills,
    }
    return fills, meta


def inject_cascade_liquidation(fills: list, cfg: CascadeLiquidationConfig,
                                coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """
    Simulate a liquidation waterfall: multiple waves of forced selling
    that drive prices progressively lower.
    """
    rng = random.Random(cfg.seed)
    fills = [copy.deepcopy(f) for f in fills]
    if not fills:
        return fills, {}

    coins = coins or list({f["coin"] for f in fills})
    target_coin = rng.choice(coins)

    t_min = fills[0]["time_ms"]
    t_max = fills[-1]["time_ms"]
    t_start = int(t_min + (t_max - t_min) * cfg.inject_at_pct)

    # Reference price
    ref_price = None
    for f in fills:
        if f["coin"] == target_coin and f["time_ms"] <= t_start:
            ref_price = f["original_price"]
    if ref_price is None:
        ref_price = next((f["original_price"] for f in fills
                          if f["coin"] == target_coin), 1000.0)

    # Build cascade waves
    injected = []
    total_drop = 0.0
    wave_drop = cfg.initial_drop_pct

    for wave in range(cfg.cascade_waves):
        wave_start = t_start + wave * cfg.wave_interval_minutes * 60 * 1000
        total_drop += wave_drop
        wave_bottom = ref_price * (1 - total_drop)

        for i in range(cfg.n_fills_per_wave):
            t = wave_start + int(rng.uniform(0, cfg.wave_interval_minutes * 60 * 1000 * 0.8))
            # Price within wave: cascading downward
            fill_progress = i / max(1, cfg.n_fills_per_wave - 1)
            wave_top = ref_price * (1 - (total_drop - wave_drop))
            price = wave_top - (wave_top - wave_bottom) * fill_progress

            # Liquidation fills have worse slippage
            slippage = 0.005 * (1 + wave * 0.5)  # slippage worsens each wave

            fill = {
                "wallet_address": f"0xstress_cascade_w{wave}",
                "coin": target_coin,
                "side": "sell",
                "original_price": round(price, 6),
                "penalised_price": round(price * (1 - slippage), 6),
                "size": round(rng.uniform(0.05, 1.0) * (1 + wave * 0.3), 6),
                "time_ms": t,
                "delayed_time_ms": t + 500,  # fast liquidation
                "closed_pnl": round(-abs(rng.gauss(500, 200) * (1 + wave * 0.5)), 4),
                "penalised_pnl": 0.0,
                "fee": round(price * slippage * 0.1, 4),
                "is_liquidation": 1,
                "direction": "Close Long",
            }
            injected.append(fill)

        wave_drop *= cfg.wave_amplification

    # Recovery bounce (weak)
    t_bounce_start = t_start + cfg.cascade_waves * cfg.wave_interval_minutes * 60 * 1000
    final_bottom = ref_price * (1 - total_drop)
    bounce_price = final_bottom + (ref_price - final_bottom) * cfg.recovery_pct

    for i in range(10):
        t = t_bounce_start + int(rng.uniform(0, 30 * 60 * 1000))
        progress = i / 9.0
        price = final_bottom + (bounce_price - final_bottom) * progress
        fill = {
            "wallet_address": "0xstress_cascade_bounce",
            "coin": target_coin,
            "side": "buy",
            "original_price": round(price, 6),
            "penalised_price": round(price * 1.003, 6),
            "size": round(rng.uniform(0.02, 0.3), 6),
            "time_ms": t,
            "delayed_time_ms": t + 2000,
            "closed_pnl": 0.0,
            "penalised_pnl": 0.0,
            "fee": round(price * 0.0005, 4),
            "is_liquidation": 0,
            "direction": "Open Long",
        }
        injected.append(fill)

    # Also mutate existing fills in the cascade window
    t_end = t_bounce_start + 30 * 60 * 1000
    for f in fills:
        if f["coin"] != target_coin:
            continue
        t = f["time_ms"]
        if t < t_start or t > t_end:
            continue
        progress = min(1.0, (t - t_start) / max(1, t_end - t_start))
        crash_price = ref_price * (1 - total_drop * progress)
        f["original_price"] = round(crash_price, 6)
        slippage = 0.003 * (1 + progress * 3)
        if f["side"] == "buy":
            f["penalised_price"] = round(crash_price * (1 + slippage), 6)
        else:
            f["penalised_price"] = round(crash_price * (1 - slippage), 6)

    fills.extend(injected)
    fills.sort(key=lambda f: f["time_ms"])

    meta = {
        "coin": target_coin,
        "ref_price": ref_price,
        "total_drop_pct": round(total_drop * 100, 1),
        "final_bottom": round(ref_price * (1 - total_drop), 2),
        "waves": cfg.cascade_waves,
        "fills_injected": len(injected),
    }
    return fills, meta


def inject_black_swan(fills: list, cfg: BlackSwanConfig,
                       coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """
    Combined scenario: flash crash + funding squeeze + liquidity drain +
    cascade liquidation all firing within a tight window.
    """
    meta = {"components": {}}

    # 1. Flash crash (amplified)
    fc = FlashCrashConfig(
        drop_pct=0.25 * cfg.crash_severity,
        duration_minutes=8,
        panic_slippage_mult=8.0 * cfg.crash_severity,
        n_crash_fills=60,
        inject_at_pct=cfg.inject_at_pct,
        seed=cfg.seed,
    )
    fills, fc_meta = inject_flash_crash(fills, fc, coins)
    meta["components"]["flash_crash"] = fc_meta

    # 2. Funding squeeze (amplified, starts slightly later)
    fs = FundingSqueezeConfig(
        peak_funding_rate_8h=0.015 * cfg.funding_severity,
        squeeze_duration_hours=cfg.window_hours,
        price_drift_pct=-0.12,
        inject_at_pct=min(0.95, cfg.inject_at_pct + 0.01),
        seed=cfg.seed + 1,
    )
    fills, fs_meta = inject_funding_squeeze(fills, fs, coins)
    meta["components"]["funding_squeeze"] = fs_meta

    # 3. Liquidity drain (amplified)
    ld = LiquidityDrainConfig(
        crisis_slippage_bps=150 * cfg.liquidity_severity,
        drain_duration_hours=cfg.window_hours,
        partial_fill_rate=0.80,
        inject_at_pct=cfg.inject_at_pct,
        seed=cfg.seed + 2,
    )
    fills, ld_meta = inject_liquidity_drain(fills, ld, coins)
    meta["components"]["liquidity_drain"] = ld_meta

    # 4. Cascade liquidation (amplified)
    cl = CascadeLiquidationConfig(
        initial_drop_pct=0.10,
        cascade_waves=int(4 * cfg.liquidation_severity),
        wave_amplification=1.5,
        inject_at_pct=min(0.95, cfg.inject_at_pct + 0.02),
        seed=cfg.seed + 3,
    )
    fills, cl_meta = inject_cascade_liquidation(fills, cl, coins)
    meta["components"]["cascade_liquidation"] = cl_meta

    meta["window_hours"] = cfg.window_hours
    return fills, meta


# ─── Dispatcher ───────────────────────────────────────────────────

INJECTORS = {
    "flash_crash": inject_flash_crash,
    "funding_squeeze": inject_funding_squeeze,
    "liquidity_drain": inject_liquidity_drain,
    "cascade_liquidation": inject_cascade_liquidation,
    "black_swan": inject_black_swan,
}


def apply_scenario(fills: list, scenario_key: str, scenario_cfg,
                    coins: Optional[List[str]] = None) -> Tuple[list, Dict]:
    """Apply a named scenario to a fill list. Returns (mutated_fills, metadata)."""
    injector = INJECTORS.get(scenario_key)
    if injector is None:
        raise ValueError(f"Unknown scenario: {scenario_key}")
    return injector(fills, scenario_cfg, coins)
