"""A5: Statistical promotion-gate primitives — Deflated Sharpe + SPRT.

The shadow→canary→live promotion pipeline already exists (see
:mod:`src.learning.shadow_evaluator`, :mod:`src.learning.promotion_gate`).
What's missing is statistical rigor on "is the challenger actually
better?" — without it, multi-testing bias and small-sample noise mean
overfit candidates leak through promotion criteria.

Two primitives here:

1. ``deflated_sharpe(returns, num_trials, skew, kurt)`` —
   Bailey & López de Prado's Deflated Sharpe Ratio (SSRN 2460551),
   which corrects the Sharpe estimate for (a) the number of trials
   behind a backtest's max-SR and (b) the higher-moment risk of the
   return distribution. This is the canonical defense against the
   "lucky-backtest" failure mode of any continuous-learning loop.

2. ``sprt_pair(challenger, champion, alpha, beta, mde)`` —
   Wald's Sequential Probability Ratio Test, paired against an
   incumbent baseline. Returns ACCEPT / REJECT / CONTINUE so the
   promotion loop can stop as soon as evidence is conclusive without
   pre-committing to a sample size.

Both functions are *pure* — no DB or filesystem dependencies — so
they're trivially testable and reusable.

References
----------
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-
  Normality." SSRN 2460551.
- Wald (1947), Sequential Analysis. Wiley.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


# Euler-Mascheroni constant — used in the expected-max-Sharpe formula.
_EULER_GAMMA = 0.5772156649015329


# ── Helpers ────────────────────────────────────────────────────────────


def _to_list(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs]


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _variance(xs: Sequence[float], mean: Optional[float] = None) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean if mean is not None else _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _stddev(xs: Sequence[float], mean: Optional[float] = None) -> float:
    return math.sqrt(_variance(xs, mean=mean))


def _skewness(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = _stddev(xs, mean=m)
    if s <= 0:
        return 0.0
    return sum((x - m) ** 3 for x in xs) / (n * s**3)


def _kurtosis(xs: Sequence[float]) -> float:
    """Excess-kurtosis NOT used here; we return the *raw* kurtosis used
    in the Bailey-LdP DSR formula (normal-distribution baseline = 3).
    """
    n = len(xs)
    if n < 4:
        return 3.0
    m = _mean(xs)
    s2 = _variance(xs, mean=m)
    if s2 <= 0:
        return 3.0
    return sum((x - m) ** 4 for x in xs) / (n * s2**2)


def _standard_normal_cdf(x: float) -> float:
    """Φ(x) using the error function — math.erf is available in stdlib."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _standard_normal_inv_cdf(p: float) -> float:
    """Inverse Φ⁻¹(p) via the rational Beasley-Springer-Moro algorithm.

    Sufficient for promotion-gate use (we need z at α = 0.05/0.01-level
    typical tail-probability lookups, not high-precision tail
    integration). Clamps p to (0, 1) to avoid math domain errors.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    # Coefficients from Acklam (2000); accuracy ~ 1e-9 over the central
    # region. Good enough for promotion thresholds.
    a = [
        -39.69683028665376, 220.9460984245205, -275.9285104469687,
        138.3577518672690, -30.66479806614716, 2.506628277459239,
    ]
    b = [
        -54.47609879822406, 161.5858368580409, -155.6989798598866,
        66.80131188771972, -13.28068155288572,
    ]
    c = [
        -0.007784894002430293, -0.3223964580411365, -2.400758277161838,
        -2.549732539343734, 4.374664141464968, 2.938163982698783,
    ]
    d = [
        0.007784695709041462, 0.3224671290700398, 2.445134137142996,
        3.754408661907416,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────


@dataclass
class DSRResult:
    sharpe: float
    deflated_sharpe: float
    p_value: float
    significant_at_95: bool
    num_observations: int
    num_trials: int


def sharpe_ratio(returns: Sequence[float]) -> float:
    """Annualisation-agnostic raw Sharpe ratio: mean / stddev.

    Caller multiplies by sqrt(periods_per_year) externally if they want
    an annualised number. DSR uses the same scaling consistently so
    significance is independent of annualisation choice.
    """
    if len(returns) <= 1:
        return 0.0
    mu = _mean(returns)
    sigma = _stddev(returns, mean=mu)
    if sigma <= 0:
        return 0.0
    return mu / sigma


def deflated_sharpe(
    returns: Sequence[float],
    *,
    num_trials: int,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio and its significance.

    Parameters
    ----------
    returns
        Per-period returns of the candidate strategy.
    num_trials
        Total number of strategies / hyperparameter configurations
        tried during candidate discovery. THE PARAMETER. If you ran
        the backtest harness 50 times before picking this candidate,
        num_trials=50 — even if 49 of those runs were on previous
        days. Promotion code must track this.
    skew, kurt
        Sample skewness and (raw, non-excess) kurtosis. Computed from
        ``returns`` if not provided. Pass explicitly when you have a
        better estimate from a longer history.

    Returns
    -------
    DSRResult with the raw Sharpe, deflated Sharpe (under the null
    hypothesis the Sharpe is approximately standard-normal), and a
    convenience boolean for significance at 95% confidence.

    Notes
    -----
    Implementation follows Bailey & López de Prado eq. (4)–(6) in
    "The Deflated Sharpe Ratio" (SSRN 2460551). The expected max
    Sharpe across N trials is approximated as:

        SR0 ≈ sqrt(Var[SR_n]) * ((1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e)))

    where γ is the Euler-Mascheroni constant. The deflated Sharpe is
    then:

        DSR = ((SR - SR0) * sqrt(T-1))
              / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)

    and is approximately N(0, 1) under the null.
    """
    rs = _to_list(returns)
    n_obs = len(rs)
    if n_obs <= 1:
        return DSRResult(
            sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            significant_at_95=False, num_observations=n_obs,
            num_trials=int(num_trials),
        )
    num_trials = max(1, int(num_trials))

    sr = sharpe_ratio(rs)
    s = float(skew if skew is not None else _skewness(rs))
    k = float(kurt if kurt is not None else _kurtosis(rs))

    # Expected max Sharpe under the null across N trials. Bailey-LdP
    # eq. (5): the extreme-value factor scales the standard deviation
    # of the single-trial Sharpe-estimator, which for T observations
    # under the null is approximately 1/sqrt(T-1). So SR0 in
    # sample-Sharpe units (matching `sr`) is the factor divided by
    # sqrt(T-1).
    sqrt_t_minus_one = math.sqrt(max(n_obs - 1, 1))
    if num_trials == 1:
        sr0 = 0.0
    else:
        z_a = _standard_normal_inv_cdf(1.0 - 1.0 / num_trials)
        z_b = _standard_normal_inv_cdf(1.0 - 1.0 / (num_trials * math.e))
        extreme_value_factor = (1.0 - _EULER_GAMMA) * z_a + _EULER_GAMMA * z_b
        sr0 = extreme_value_factor / sqrt_t_minus_one

    # Variance of Sharpe estimate accounting for higher moments
    var_term = 1.0 - s * sr + ((k - 1.0) / 4.0) * sr**2
    if var_term <= 0:
        # Pathological distribution — return a non-significant result
        # rather than crash the promotion loop.
        return DSRResult(
            sharpe=sr, deflated_sharpe=0.0, p_value=1.0,
            significant_at_95=False, num_observations=n_obs,
            num_trials=num_trials,
        )

    dsr_z = (sr - sr0) * sqrt_t_minus_one / math.sqrt(var_term)
    p_value = 1.0 - _standard_normal_cdf(dsr_z)
    return DSRResult(
        sharpe=sr,
        deflated_sharpe=dsr_z,
        p_value=p_value,
        significant_at_95=(p_value < 0.05),
        num_observations=n_obs,
        num_trials=num_trials,
    )


# ── Sequential Probability Ratio Test ─────────────────────────────────


@dataclass
class SPRTResult:
    decision: str           # "ACCEPT" | "REJECT" | "CONTINUE"
    log_likelihood_ratio: float
    upper_threshold: float
    lower_threshold: float
    num_observations: int


def sprt_pair(
    challenger_returns: Sequence[float],
    champion_returns: Sequence[float],
    *,
    alpha: float = 0.05,
    beta: float = 0.05,
    mde: float = 0.01,
) -> SPRTResult:
    """Paired SPRT comparing challenger vs. champion per-period returns.

    Tests H0: mean(challenger - champion) == 0  (no improvement)
    vs.    H1: mean(challenger - champion) >= mde (minimum detectable edge)

    Parameters
    ----------
    challenger_returns, champion_returns
        Per-period returns of the challenger and champion strategies,
        aligned period-by-period. The function pairs them by index.
    alpha
        Type-I error rate (false ACCEPT of challenger when no edge).
    beta
        Type-II error rate (false REJECT of a truly better challenger).
    mde
        Minimum detectable effect — the per-period excess return that
        constitutes "actually better." Pick this from operational
        cost-of-switching reasoning: if you have to slow live trades to
        canary a new model, mde = (cost / expected_trade_count) is a
        sane lower bound.

    Returns
    -------
    SPRTResult with the running log-likelihood ratio and a decision
    string. ACCEPT means challenger beats champion at the configured
    edge with the configured error rates; REJECT means evidence is
    against; CONTINUE means insufficient evidence either way and the
    caller should collect more samples.

    Notes
    -----
    Under H1 the per-period diff is modeled as N(mde, σ̂²) where σ̂² is
    the sample variance of the diff series — that's the standard
    Wald-SPRT formulation for shift-in-mean detection. With a 0 mean
    under H0 and an mde mean under H1 the LLR contribution per
    observation simplifies to:

        llr_i = (diff_i - mde/2) * (mde / σ̂²)

    and we accumulate Σ llr_i, comparing against
    log((1-β)/α) (ACCEPT) and log(β/(1-α)) (REJECT).
    """
    a = _to_list(challenger_returns)
    b = _to_list(champion_returns)
    n = min(len(a), len(b))
    if n < 2:
        # Not enough samples to even estimate variance — defer.
        return SPRTResult(
            decision="CONTINUE",
            log_likelihood_ratio=0.0,
            upper_threshold=math.log((1 - beta) / alpha) if alpha > 0 else float("inf"),
            lower_threshold=math.log(beta / (1 - alpha)) if alpha < 1 else float("-inf"),
            num_observations=n,
        )
    diffs = [a[i] - b[i] for i in range(n)]
    mu_diff = _mean(diffs)
    var_diff = _variance(diffs, mean=mu_diff)
    if var_diff <= 0:
        # Identical returns — no information, defer.
        return SPRTResult(
            decision="CONTINUE",
            log_likelihood_ratio=0.0,
            upper_threshold=math.log((1 - beta) / alpha) if alpha > 0 else float("inf"),
            lower_threshold=math.log(beta / (1 - alpha)) if alpha < 1 else float("-inf"),
            num_observations=n,
        )

    # Wald log-likelihood ratio for shift-in-mean with known variance
    # (estimated from the diff sample).
    llr = sum((d - mde / 2.0) * (mde / var_diff) for d in diffs)
    upper = math.log((1 - beta) / alpha) if 0 < alpha < 1 else float("inf")
    lower = math.log(beta / (1 - alpha)) if 0 < alpha < 1 else float("-inf")
    if llr >= upper:
        decision = "ACCEPT"
    elif llr <= lower:
        decision = "REJECT"
    else:
        decision = "CONTINUE"
    return SPRTResult(
        decision=decision,
        log_likelihood_ratio=llr,
        upper_threshold=upper,
        lower_threshold=lower,
        num_observations=n,
    )
