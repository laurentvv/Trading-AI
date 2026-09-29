"""Métriques de performance et tests de robustesse (bootstrap, Sharpe déflaté).

Sharpe/Sortino sans taux sans risque (le cash T212 est peu rémunéré ; hypothèse à durcir plus tard).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .engine import BacktestResult

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def sharpe(returns: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods)) if sd > 0 else 0.0


def sortino(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    downside = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / downside * math.sqrt(periods)) if downside > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Pire repli depuis un sommet, en valeur négative (-0,25 = -25 %)."""
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def compute_metrics(result: BacktestResult) -> dict:
    eq, tr = result.equity, result.trades
    rets = daily_returns(eq)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    mdd = max_drawdown(eq)
    c = cagr(eq)
    sells = tr[tr["side"] == "SELL"]
    wins, losses = sells[sells["return_pct"] > 0], sells[sells["return_pct"] <= 0]
    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "cagr": c,
        "vol": float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0,
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": mdd,
        "calmar": float(c / abs(mdd)) if mdd < 0 else 0.0,
        "time_invested": float((result.exposure > 0.01).mean()),
        "avg_exposure": float(result.exposure.mean()),
        "turnover_per_year": float(tr["notional"].sum() / eq.mean() / years),
        "n_trades": int(len(sells)),
        "win_rate": float(len(wins) / len(sells)) if len(sells) else 0.0,
        "avg_win": float(wins["return_pct"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["return_pct"].mean()) if len(losses) else 0.0,
        "tax_paid": float(tr["tax"].sum()),
    }


def paired_block_bootstrap_sharpe_diff(
    returns_a: pd.Series, returns_b: pd.Series, n_boot: int = 2000, block: int = 21, seed: int = 0
) -> dict:
    """IC à 95 % de Sharpe(A) - Sharpe(B) par bootstrap circulaire PAR BLOCS (préserve l'autocorrélation),
    en rééchantillonnant les mêmes jours pour A et B. `p_le_0` = part des tirages où A n'a pas battu B."""
    a, b = returns_a.align(returns_b, join="inner")
    a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    n = len(a)
    if n < 2 * block:
        raise ValueError("série trop courte pour ce bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    diffs = np.empty(n_boot)
    offsets = np.arange(block)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        diffs[k] = sharpe(a[idx]) - sharpe(b[idx])
    return {
        "diff": sharpe(a) - sharpe(b),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_le_0": float((diffs <= 0).mean()),
    }


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes_annual: list[float]) -> float:
    """Probabilité que le vrai Sharpe soit > 0 une fois corrigé du nombre de variantes testées
    (Bailey & López de Prado, 2014). `trial_sharpes_annual` = Sharpe annualisé de TOUTES les variantes essayées."""
    r = np.asarray(returns, dtype=float)
    t = len(r)
    n_trials = len(trial_sharpes_annual)
    if t < 3 or n_trials < 2 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)  # par période
    trial_sr = np.asarray(trial_sharpes_annual, dtype=float) / math.sqrt(TRADING_DAYS)
    euler = 0.5772156649
    sr0 = trial_sr.std(ddof=1) * (
        (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials) + euler * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    )
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom))
