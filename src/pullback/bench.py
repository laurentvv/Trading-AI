"""Banc de la méthode pullback : références, variantes pré-enregistrées, contrôles, walk-forward, verdict.

Usage :  python -m src.pullback.bench [--limit N] [--out docs/BACKTEST_PULLBACK_<date>.md]

Calibrage des variantes sur 2015-2020, TEST sur 2021 -> fin des données (aucun réglage sur le test).
Tout est net de frais et d'impôt. Le verdict de la porte du plan (§2.1) est calculé ici, jamais écrit à la main.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import metrics as bm
from src.backtest import strategies
from src.backtest.engine import run_backtest
from src.pullback import market, portfolio, rules, store

logger = logging.getLogger("pullback.bench")

CACHE_DIR = Path("data_cache")
REFS_DIR = CACHE_DIR / "pullback_refs"
CAL_START, CAL_END = "2015-01-02", "2020-12-31"
TEST_START = "2021-01-04"
SIGNALS_FROM = "2014-12-01"
CAPITAL = 30_000.0
COST_BPS = 25.0  # change 0,15 % + 10 pb de spread/glissement, par sens
ETF_COST_BPS = 10.0  # ETF en EUR : pas de change
TAX = 0.30
N_RANDOM = 40
LARGE_CAP_DOLLAR_VOLUME = 200_000_000.0


@dataclass(frozen=True)
class Variant:
    name: str
    ema_mode: str = "always"
    rank80: bool = False
    atr_max: float = 0.8
    max_dist: float = 0.03
    market_filter: bool = False


# Variantes PRÉ-ENREGISTRÉES (plan §1) : la liste est figée avant d'avoir vu un résultat ; son nombre
# alimente le Sharpe déflaté. Non testées : stop remonté à l'entrée après 3R, allègement sur extension.
VARIANTS = [
    Variant("base"),
    Variant("ema9_apres_3R", ema_mode="after_3r"),
    Variant("rang_80", rank80=True),
    Variant("atr_0.7", atr_max=0.7),
    Variant("stop_4pct", max_dist=0.04),
    Variant("filtre_marche_QQQ", market_filter=True),
]


# --- données de référence --------------------------------------------------------------------------------


def _download_ref(sym: str) -> pd.DataFrame:
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    f = REFS_DIR / f"{sym.replace('=', '_').replace('^', '_')}.parquet"
    if f.exists() and (pd.Timestamp.today() - pd.Timestamp(f.stat().st_mtime, unit="s")).days < 1:
        return pd.read_parquet(f)
    import yfinance as yf

    raw = yf.download(sym, start="2013-01-01", interval="1d", auto_adjust=True, progress=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError(f"référence {sym} introuvable")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    raw = raw.dropna(subset=["Open", "Close"])
    raw = raw[(raw["Open"] > 0) & (raw["Close"] > 0)]
    raw.to_parquet(f)
    return raw


def load_refs(dates: pd.DatetimeIndex) -> dict:
    fx = _download_ref("EURUSD=X")["Close"].reindex(dates).ffill().bfill()
    return {
        "fx": fx.to_numpy(dtype=float),  # USD par EUR
        "ndx": _download_ref("SXRV.DE"),
        "spx": _download_ref("SXR8.DE"),
        "qqq": _download_ref("QQQ"),
    }


def reference_runs(prices: pd.DataFrame, lo: str, hi: str) -> dict[str, tuple[pd.Series, dict]]:
    """B&H et MA200 d'un ETF EUR sur [lo, hi], nets de frais et d'impôt (cible calculée sur tout l'historique)."""
    out = {}
    targets = {"B&H": strategies.buy_and_hold(prices), "MA200": strategies.ma_trend(prices, 200, 0.0)}
    sl = prices.loc[lo:hi]
    for name, tgt in targets.items():
        res = run_backtest(
            sl, tgt.reindex(sl.index), initial_capital=CAPITAL, cost_bps=ETF_COST_BPS, tax_rate=TAX, liquidate_at_end=True
        )
        out[name] = (res.equity, bm.compute_metrics(res))
    return out


# --- signaux -> ordres -----------------------------------------------------------------------------------


def build_orders(sig: pd.DataFrame, v: Variant, market_ok: np.ndarray | None, T: int) -> dict[int, list]:
    s = sig[(sig["distance"] <= v.max_dist) & (sig["atr_ratio"] < v.atr_max)]
    if v.rank80:
        s = s[s["pass80"]]
    if v.market_filter and market_ok is not None:
        s = s[market_ok[s["t"].to_numpy()]]
    orders: dict[int, list] = {}
    for t, j, trig, stop, prio in zip(s["t"], s["j"], s["trigger"], s["stop"], s["priority"]):
        if t + 1 < T:
            orders.setdefault(int(t) + 1, []).append((int(j), float(trig), float(stop), float(prio)))
    return orders


def control_orders(
    mkt: market.Market,
    liquid: np.ndarray,
    v: Variant,
    market_ok: np.ndarray | None,
    counts: dict[int, int],
    *,
    mode: str,
    seed: int,
    ranks: tuple,
) -> dict[int, list]:
    """Contrôles : même nombre d'ordres par jour que la stratégie, mais candidats = titres momentum de la
    journée (rang >= seuil) dont l'amplitude du jour tient dans la distance de stop. `random` : tirage au
    hasard ; `momentum` : les k meilleurs rangs à 63 séances. Aucun test de consolidation ni de support."""
    sel, sel80, rank63 = ranks
    base = sel80 if v.rank80 else sel
    rng = np.random.default_rng(seed)
    orders: dict[int, list] = {}
    T = len(mkt.dates)
    for t, k in counts.items():
        if t + 1 >= T or (v.market_filter and market_ok is not None and not market_ok[t]):
            continue
        h, lo = mkt.H[t], mkt.L[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = base[t] & np.isfinite(h) & (h > 0) & ((h - lo) / h <= v.max_dist) & ((h - lo) > 0)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        if mode == "random":
            pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
            prio = rng.random(len(pick))
        else:
            order = idx[np.argsort(-np.nan_to_num(rank63[t, idx]))][:k]
            pick, prio = order, np.arange(len(order), 0, -1, dtype=float)
        orders[t + 1] = [(int(j), float(h[j]), float(lo[j]), float(pr)) for j, pr in zip(pick, prio)]
    return orders


# --- métriques ------------------------------------------------------------------------------------------


def equity_metrics(eq: pd.Series, tr: pd.DataFrame, exposure: pd.Series | None = None) -> dict:
    r = bm.daily_returns(eq)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    wins, losses = tr[tr["r"] > 0], tr[tr["r"] <= 0]
    return {
        "final": float(eq.iloc[-1]),
        "cagr": bm.cagr(eq),
        "sharpe": bm.sharpe(r),
        "max_dd": bm.max_drawdown(eq),
        "n_trades": int(len(tr)),
        "trades_per_year": float(len(tr) / years),
        "win_rate": float(len(wins) / len(tr)) if len(tr) else 0.0,
        "expectancy_r": float(tr["r"].mean()) if len(tr) else 0.0,
        "avg_win_r": float(wins["r"].mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses["r"].mean()) if len(losses) else 0.0,
        "avg_bars": float(tr["bars"].mean()) if len(tr) else 0.0,
        "avg_exposure": float(exposure.mean()) if exposure is not None else float("nan"),
    }


def yearly_returns(eq: pd.Series) -> dict[int, float]:
    out = {}
    for y, g in eq.groupby(eq.index.year):
        prev = eq[eq.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else g.iloc[0]
        out[int(y)] = float(g.iloc[-1] / base - 1.0)
    return out


# --- exécution du banc ---------------------------------------------------------------------------------


class Bench:
    def __init__(self, mkt: market.Market, refs: dict, sig: pd.DataFrame):
        self.mkt, self.refs, self.sig = mkt, refs, sig
        self.T = len(mkt.dates)
        q = refs["qqq"]["Close"].reindex(mkt.dates).ffill()
        self.market_ok = (q > rules.ema(q, 21)).fillna(False).to_numpy()
        self.liquid = mkt.liquid()
        self._ranks = None

    def idx(self, lo: str, hi: str) -> tuple[int, int]:
        a = int(self.mkt.dates.searchsorted(pd.Timestamp(lo)))
        b = int(self.mkt.dates.searchsorted(pd.Timestamp(hi), side="right")) - 1
        return a, b

    def run(
        self, orders: dict, lo: str, hi: str, v: Variant, *, cost_bps: float = COST_BPS, intraday: str = "worst"
    ) -> portfolio.PortfolioResult:
        a, b = self.idx(lo, hi)
        m = self.mkt
        return portfolio.run_portfolio(
            m.dates, m.O, m.H, m.L, m.C, m.ema9, self.refs["fx"], orders, m.symbols,
            start=a, end=b, capital=CAPITAL, cost_bps=cost_bps, tax_rate=TAX, ema_mode=v.ema_mode,
            intraday=intraday,
        )  # fmt: skip

    def ranks(self):
        if self._ranks is None:
            self._ranks = market.momentum_masks(self.mkt, self.liquid)
        return self._ranks

    def evaluate(self, v: Variant, lo: str, hi: str, **kw) -> tuple[dict, portfolio.PortfolioResult]:
        res = self.run(build_orders(self.sig, v, self.market_ok, self.T), lo, hi, v, **kw)
        m = equity_metrics(res.equity, res.trades, res.exposure)
        m["fees_eur"], m["tax_eur"] = res.fees_eur, res.tax_eur
        return m, res


def gate(test: dict, ndx: dict, rand_p: float, rand_p_exp: float, large: dict) -> list[dict]:
    """Porte du plan §2.1 : chaque critère avec sa valeur, son seuil et son résultat."""
    rows = [
        ("CAGR > buy & hold Nasdaq-100 EUR", test["cagr"], ndx["cagr"], test["cagr"] > ndx["cagr"]),
        ("Sharpe > buy & hold Nasdaq-100 EUR", test["sharpe"], ndx["sharpe"], test["sharpe"] > ndx["sharpe"]),
        ("Drawdown maximal >= -35 %", test["max_dd"], -0.35, test["max_dd"] >= -0.35),
        ("Espérance >= +0,25 R", test["expectancy_r"], 0.25, test["expectancy_r"] >= 0.25),
        ("Au moins 150 trades", test["n_trades"], 150, test["n_trades"] >= 150),
        ("Contrôle « entrées au hasard » battu (Sharpe, p <= 0,05)", rand_p, 0.05, rand_p <= 0.05),
        ("Contrôle « entrées au hasard » battu (espérance R, p <= 0,05)", rand_p_exp, 0.05, rand_p_exp <= 0.05),
        (
            "Positif sur le sous-univers très liquide (proxy du biais du survivant)",
            large["cagr"],
            0.0,
            large["cagr"] > 0 and large["expectancy_r"] > 0 and large["n_trades"] >= 30,
        ),
    ]
    return [{"critere": n, "valeur": v, "seuil": s, "ok": bool(ok)} for n, v, s, ok in rows]


def _pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f} %"


def _row(name: str, m: dict) -> list[str]:
    return [
        name, _pct(m["cagr"]), f"{m['sharpe']:.2f}", _pct(m["max_dd"]), str(m.get("n_trades", "")),
        _pct(m.get("win_rate", float("nan"))) if m.get("n_trades") else "", f"{m.get('expectancy_r', float('nan')):+.2f}" if m.get("n_trades") else "",
    ]  # fmt: skip


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ref_metrics(eq: pd.Series, met: dict) -> dict:
    return {"cagr": met["cagr"], "sharpe": met["sharpe"], "max_dd": met["max_drawdown"]}


def run_bench(mkt: market.Market, refs: dict, sig: pd.DataFrame, sig_large: pd.DataFrame, out_md: Path | None) -> dict:
    B = Bench(mkt, refs, sig)
    last = str(mkt.dates[-1].date())
    cal_hi = CAL_END
    # 1) variantes sur calibrage et test
    cal: dict[str, dict] = {}
    test: dict[str, dict] = {}
    results = {}
    for v in VARIANTS:
        cal[v.name], _ = B.evaluate(v, CAL_START, cal_hi)
        test[v.name], results[v.name] = B.evaluate(v, TEST_START, last)
        logger.info("%s cal Sharpe %.2f | test Sharpe %.2f", v.name, cal[v.name]["sharpe"], test[v.name]["sharpe"])
    best = max(VARIANTS, key=lambda v: cal[v.name]["sharpe"])
    # 2) références (net de frais/impôt) sur les mêmes fenêtres
    refs_cal = {k: reference_runs(refs[k], CAL_START, cal_hi) for k in ("ndx", "spx")}
    refs_test = {k: reference_runs(refs[k], TEST_START, last) for k in ("ndx", "spx")}
    ndx_bh = ref_metrics(*refs_test["ndx"]["B&H"])
    # 3) contrôles sur le test, variante retenue
    sel_orders = build_orders(sig, best, B.market_ok, B.T)
    counts = {t - 1: len(o) for t, o in sel_orders.items()}
    a, b = B.idx(TEST_START, last)
    counts = {t: k for t, k in counts.items() if a - 1 <= t <= b - 1}
    ranks = B.ranks()
    strat = test[best.name]
    rand_m = []
    for seed in range(N_RANDOM):
        o = control_orders(mkt, B.liquid, best, B.market_ok, counts, mode="random", seed=seed, ranks=ranks)
        res = B.run(o, TEST_START, last, best)
        rand_m.append(equity_metrics(res.equity, res.trades, res.exposure))
    o = control_orders(mkt, B.liquid, best, B.market_ok, counts, mode="momentum", seed=0, ranks=ranks)
    mom_res = B.run(o, TEST_START, last, best)
    mom_m = equity_metrics(mom_res.equity, mom_res.trades, mom_res.exposure)
    p_sharpe = (1 + sum(m["sharpe"] >= strat["sharpe"] for m in rand_m)) / (1 + N_RANDOM)
    p_exp = (1 + sum(m["expectancy_r"] >= strat["expectancy_r"] for m in rand_m)) / (1 + N_RANDOM)
    # 4) sous-univers très liquide (proxy partiel du biais du survivant)
    Bl = Bench(mkt, refs, sig_large)
    Bl.liquid = mkt.liquid(min_dollar_volume=LARGE_CAP_DOLLAR_VOLUME)
    large, _ = Bl.evaluate(best, TEST_START, last)
    # 5) robustesse : coûts, bootstrap vs B&H, Sharpe déflaté
    cost_sens = {c: B.evaluate(best, TEST_START, last, cost_bps=c)[0] for c in (0.0, 10.0, 25.0, 40.0)}
    # sensibilité à l'ordre intrajournalier supposé (le verdict reste celui de "worst", règle du plan)
    intraday_sens = {}
    for mode in ("worst", "ohlc", "best"):
        m_s, _ = B.evaluate(best, TEST_START, last, intraday=mode)
        rr = []
        for seed in range(20):
            o = control_orders(mkt, B.liquid, best, B.market_ok, counts, mode="random", seed=seed, ranks=ranks)
            rr.append(equity_metrics(*(lambda r: (r.equity, r.trades, r.exposure))(B.run(o, TEST_START, last, best, intraday=mode))))
        intraday_sens[mode] = (m_s, {k: float(pd.DataFrame(rr)[k].median()) for k in rr[0]})
    r_strat = bm.daily_returns(results[best.name].equity)
    r_ndx = bm.daily_returns(refs_test["ndx"]["B&H"][0])
    boot = bm.paired_block_bootstrap_sharpe_diff(r_strat, r_ndx)
    dsr = bm.deflated_sharpe_ratio(r_strat, [cal[v.name]["sharpe"] for v in VARIANTS])
    g = gate(strat, ndx_bh, p_sharpe, p_exp, large)
    verdict = all(r["ok"] for r in g)
    out = {
        "best": best, "cal": cal, "test": test, "refs_cal": refs_cal, "refs_test": refs_test, "rand": rand_m,
        "momentum": mom_m, "p_sharpe": p_sharpe, "p_exp": p_exp, "large": large, "cost_sens": cost_sens,
        "boot": boot, "dsr": dsr, "intraday": intraday_sens, "gate": g, "verdict": verdict, "results": results, "last": last,
        "n_symbols": len(mkt.symbols), "n_signals": len(sig),
    }  # fmt: skip
    if out_md is not None:
        out_md.write_text(render(out), encoding="utf-8")
    return out


def render(o: dict) -> str:
    best, last = o["best"], o["last"]
    st = o["test"][best.name]
    ndx_t, spx_t = o["refs_test"]["ndx"], o["refs_test"]["spx"]
    rand = pd.DataFrame(o["rand"])
    L = [
        "# Banc de la méthode pullback (généré par `src.pullback.bench`)",
        "",
        f"**Données jusqu'au {last}.** {o['n_symbols']} symboles US de Trading 212 avec des barres valides, "
        f"{o['n_signals']} setups candidats. Capital {CAPITAL:,.0f} €, frais {COST_BPS:.0f} pb par sens "
        f"(change 0,15 % + spread/glissement), impôt {TAX * 100:.0f} % sur les plus-values nettes de l'année, "
        "équité en EUR, positions soldées à la fin pour comparer honnêtement au buy & hold.",
        "",
        "## Verdict de la porte du plan (§2.1)",
        "",
        f"### {'PASS (provisoire)' if o['verdict'] else 'FAIL'}",
        "",
        _table(
            ["Critère", "Valeur", "Seuil", "OK"],
            [
                [r["critere"], f"{r['valeur']:.3f}" if isinstance(r["valeur"], float) else str(r["valeur"]),
                 f"{r['seuil']:.3f}" if isinstance(r["seuil"], float) else str(r["seuil"]), "oui" if r["ok"] else "**non**"]
                for r in o["gate"]
            ],
        ),  # fmt: skip
        "",
        "Un PASS reste **provisoire** : l'univers ne contient que des titres encore cotés aujourd'hui (les faillites "
        "et radiations ont disparu), donc les résultats absolus sont optimistes. Le contrôle « entrées au hasard » "
        "partage ce biais : l'écart stratégie - hasard est la mesure la plus fiable. Le sous-univers très liquide "
        f"(>= {LARGE_CAP_DOLLAR_VOLUME / 1e6:.0f} M$/jour) n'est qu'un proxy, pas l'univers à la date.",
        "",
        f"Variante retenue sur le calibrage {CAL_START[:4]}-{CAL_END[:4]} : **{best.name}**. "
        f"Test {TEST_START} -> {last}, sans réglage sur cette fenêtre.",
        "",
        "## Test hors échantillon",
        "",
        _table(
            ["Stratégie", "CAGR", "Sharpe", "Drawdown max", "Trades", "Réussite", "Espérance R"],
            [
                _row(f"Pullback ({best.name})", st),
                _row("B&H Nasdaq-100 EUR (SXRV.DE)", ref_metrics(*ndx_t["B&H"])),
                _row("MA200 Nasdaq-100 EUR", ref_metrics(*ndx_t["MA200"])),
                _row("B&H S&P 500 EUR (SXR8.DE)", ref_metrics(*spx_t["B&H"])),
                _row("Momentum seul (mêmes sorties, sans setup)", o["momentum"]),
                _row(f"Entrées au hasard (médiane de {N_RANDOM})", {k: float(rand[k].median()) for k in rand.columns}),
            ],
        ),
        "",
        f"- Trades par an : {st['trades_per_year']:.0f} ; durée moyenne {st['avg_bars']:.1f} séances ; exposition moyenne "
        f"{_pct(st['avg_exposure'])} ; gain moyen {st['avg_win_r']:+.2f} R, perte moyenne {st['avg_loss_r']:+.2f} R.",
        f"- Frais payés : {st['fees_eur']:,.0f} € ; impôt payé : {abs(st['tax_eur']):,.0f} €.",
        f"- Contrôle au hasard ({N_RANDOM} tirages) : Sharpe médian {rand['sharpe'].median():.2f} "
        f"(min {rand['sharpe'].min():.2f}, max {rand['sharpe'].max():.2f}) ; p(Sharpe) = {o['p_sharpe']:.3f}, "
        f"p(espérance) = {o['p_exp']:.3f}.",
        f"- Bootstrap par blocs, Sharpe(stratégie) - Sharpe(B&H Nasdaq-100) : {o['boot']['diff']:+.2f} "
        f"[{o['boot']['ci_low']:+.2f} ; {o['boot']['ci_high']:+.2f}], p(<= 0) = {o['boot']['p_le_0']:.3f}.",
        f"- Sharpe déflaté ({len(VARIANTS)} variantes) : {o['dsr']:.3f}.",
        "",
        "### Années (test)",
        "",
        _table(
            ["Année", "Pullback", "B&H Nasdaq-100"],
            [
                [str(y), _pct(r), _pct(yearly_returns(ndx_t["B&H"][0]).get(y, float("nan")))]
                for y, r in yearly_returns(o["results"][best.name].equity).items()
            ],
        ),
        "",
        "### Sensibilité à l'ordre intrajournalier supposé (test)",
        "",
        "Quand une même barre touche plusieurs niveaux, l'ordre est inconnu. `worst` (règle du plan, utilisée pour le "
        "verdict) : entrée puis stop, stop avant cible. `ohlc` : chemin O-L-H-C si la barre clôture au-dessus de "
        "l'ouverture, sinon O-H-L-C. `best` : entrée annulée si le stop cède le jour du déclenchement, cibles avant stop.",
        "",
        _table(
            ["Hypothèse", "Pullback : CAGR", "Sharpe", "Drawdown max", "Trades", "Réussite", "Espérance R",
             "Hasard (médiane de 20) : Sharpe", "Espérance R"],
            [
                _row(mode, m) + [f"{rm['sharpe']:.2f}", f"{rm['expectancy_r']:+.2f}"]
                for mode, (m, rm) in o["intraday"].items()
            ],
        ),  # fmt: skip
        "",
        "### Sensibilité aux frais (test)",
        "",
        _table(["Frais par sens", "CAGR", "Sharpe", "Drawdown max", "Trades", "Réussite", "Espérance R"],
               [_row(f"{int(c)} pb", m) for c, m in o["cost_sens"].items()]),  # fmt: skip
        "",
        "## Variantes pré-enregistrées",
        "",
        _table(
            ["Variante", f"Calibrage {CAL_START[:4]}-{CAL_END[:4]} : CAGR", "Sharpe", "DD", "Trades", "Réussite", "Esp. R",
             "Test : CAGR", "Sharpe", "DD", "Trades"],
            [
                _row(v.name, o["cal"][v.name]) + [_pct(o["test"][v.name]["cagr"]), f"{o['test'][v.name]['sharpe']:.2f}",
                                                  _pct(o["test"][v.name]["max_dd"]), str(o["test"][v.name]["n_trades"])]
                for v in VARIANTS
            ],
        ),  # fmt: skip
        "",
        "Sélection sur le Sharpe du calibrage uniquement ; les colonnes « Test » sont informatives.",
        "",
        "## Références sur le calibrage",
        "",
        _table(
            ["Stratégie", "CAGR", "Sharpe", "Drawdown max"],
            [
                [n, _pct(m["cagr"]), f"{m['sharpe']:.2f}", _pct(m["max_dd"])]
                for n, m in (
                    ("B&H Nasdaq-100 EUR", ref_metrics(*o["refs_cal"]["ndx"]["B&H"])),
                    ("MA200 Nasdaq-100 EUR", ref_metrics(*o["refs_cal"]["ndx"]["MA200"])),
                    ("B&H S&P 500 EUR", ref_metrics(*o["refs_cal"]["spx"]["B&H"])),
                )
            ],
        ),
        "",
        "## Limites connues",
        "",
        "- **Biais du survivant** non éliminé (univers = titres encore cotés ; pas de composition historique d'indice).",
        "- Dates de résultats inconnues dans l'historique : les gaps de résultats sont payés en entier (pas de filtre).",
        "- Déclencheur journalier (le plus haut du jour de setup) au lieu de l'intraday de la source.",
        "- Ordre intrajournalier inconnu : stop avant cible, entrée avant stop (hypothèse défavorable).",
        "- Change appliqué au cours de clôture du jour ; frais de change inclus dans les 25 pb.",
        "- Barres incohérentes écartées (High/Low > 5, O/C hors [L, H], lignes gelées) ; erreurs résiduelles possibles.",
    ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--refresh-signals", action="store_true")
    args = ap.parse_args(argv)
    symbols = list(pd.read_csv("data_cache/pullback_universe.csv", keep_default_na=False)["symbol"])[: args.limit]
    mkt = market.build_market(symbols)
    logger.info("marché : %d symboles x %d séances", len(mkt.symbols), len(mkt.dates))
    refs = load_refs(mkt.dates)
    cache = CACHE_DIR / f"pullback_signals_{len(mkt.symbols)}_{mkt.dates[-1].date()}.pkl"
    if cache.exists() and not args.refresh_signals:
        sig, sig_large = pickle.loads(cache.read_bytes())
    else:
        sig = market.candidate_setups(mkt, mkt.liquid(), date_from=SIGNALS_FROM)
        sig_large = market.candidate_setups(
            mkt, mkt.liquid(min_dollar_volume=LARGE_CAP_DOLLAR_VOLUME), date_from=TEST_START
        )
        cache.write_bytes(pickle.dumps((sig, sig_large)))
    logger.info("signaux : %d (liquide) / %d (très liquide, test)", len(sig), len(sig_large))
    out = Path(args.out) if args.out else None
    res = run_bench(mkt, refs, sig, sig_large, out)
    print("VERDICT", "PASS" if res["verdict"] else "FAIL", "| variante", res["best"].name)
    for r in res["gate"]:
        print(("OK  " if r["ok"] else "NON ") + r["critere"], r["valeur"], r["seuil"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
