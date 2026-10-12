"""Moteur de portefeuille et pré-filtre vectorisé (plan pullback, contrat 14-19)."""

import numpy as np
import pandas as pd
import pytest

from src.pullback import bench, market, portfolio, rules
from tests.test_pullback_rules import synth


def _run(o, h, lo, c, orders, *, ema9=None, cost=0.0, tax=0.0, mode="always", fx=None, capital=30_000.0, liq=True, intraday="worst"):
    n = len(o)
    arr = lambda x: np.array(x, dtype=float).reshape(n, 1)  # noqa: E731
    dates = pd.bdate_range("2026-01-05", periods=n)
    e9 = arr(ema9 if ema9 is not None else [0.0] * n)
    return portfolio.run_portfolio(
        dates, arr(o), arr(h), arr(lo), arr(c), e9,
        np.ones(n) if fx is None else np.array(fx, dtype=float),
        orders, ["XXX"], start=0, end=n - 1, capital=capital, cost_bps=cost, tax_rate=tax, ema_mode=mode,
        liquidate_at_end=liq, intraday=intraday,
    )  # fmt: skip


ORDER = {1: [(0, 20.0, 19.5, 1.0)]}  # exécution jour 1 : trigger 20 (+1 cent), stop 19,5 -> R = 0,51


def test_targets_3r_5r_then_stop_on_last_third():
    #        j0    j1    j2    j3    j4    j5
    o = [19.8, 19.9, 21.0, 22.0, 22.0, 19.0]
    h = [19.9, 20.5, 21.6, 22.7, 22.1, 19.0]
    lo = [19.7, 19.8, 20.8, 21.9, 21.9, 19.0]
    c = [19.8, 20.4, 21.5, 22.6, 22.0, 19.0]
    res = _run(o, h, lo, c, ORDER, mode="after_3r", ema9=[0.0] * 6, liq=False)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    # 1/3 à 21,54 (3R), 1/3 à 22,56 (5R), 1/3 sorti au gap à l'ouverture du jour 5 (19,0 < stop 19,5)
    expected = ((21.54 - 20.01) + (22.56 - 20.01) + (19.0 - 20.01)) / 3 / 0.51
    assert t["r"] == pytest.approx(expected, rel=1e-3)
    assert t["stage"] == 2


def test_gap_below_stop_pays_the_gap():
    o = [19.8, 19.9, 19.0, 19.0]
    h = [19.9, 20.5, 19.2, 19.1]
    lo = [19.7, 19.8, 18.9, 18.9]
    c = [19.8, 20.2, 19.0, 19.0]
    res = _run(o, h, lo, c, ORDER, ema9=[0.0] * 4)
    assert res.trades.iloc[0]["r"] == pytest.approx((19.0 - 20.01) / 0.51, rel=1e-3)  # < -1R


def test_entry_and_stop_same_day_assumes_worst_case():
    o = [19.8, 19.9, 20.0]
    h = [19.9, 20.5, 20.1]
    lo = [19.7, 19.4, 19.9]  # le jour 1 touche le trigger ET le stop
    c = [19.8, 20.0, 20.0]
    res = _run(o, h, lo, c, ORDER, ema9=[0.0] * 3)
    assert res.trades.iloc[0]["r"] == pytest.approx(-1.0, rel=1e-3)


def test_no_fill_if_trigger_not_reached():
    res = _run([19.8, 19.9, 20.0], [19.9, 20.0, 20.0], [19.7, 19.8, 19.9], [19.8, 19.95, 20.0], ORDER)
    assert res.trades.empty and res.equity.iloc[-1] == pytest.approx(30_000.0)


def test_gap_up_open_fills_at_open_not_trigger():
    res = _run([19.8, 20.6, 20.6, 19.0], [19.9, 20.7, 20.7, 19.1], [19.7, 20.5, 20.5, 18.9], [19.8, 20.6, 20.6, 19.0],
               ORDER, ema9=[0.0] * 4)  # fmt: skip
    t = res.trades.iloc[0]
    assert t["r"] == pytest.approx((19.0 - 20.6) / (20.6 - 19.5), rel=1e-3)  # R mesuré sur le prix réellement payé


def test_position_size_risks_half_a_percent():
    # stop touché le jour 2 au prix du stop : la perte vaut ~0,5 % du capital (R = 0,51, qty = 150/0,51)
    o = [19.8, 19.9, 19.8, 19.8]
    h = [19.9, 20.5, 19.9, 19.9]
    lo = [19.7, 19.8, 19.4, 19.4]
    c = [19.8, 20.2, 19.6, 19.6]
    res = _run(o, h, lo, c, ORDER, ema9=[0.0] * 4)
    assert res.equity.iloc[-1] == pytest.approx(30_000.0 * (1 - 0.005), rel=2e-3)


def test_line_cap_25_percent():
    # stop très proche : le risque à 0,5 % donnerait une ligne énorme, plafonnée à 25 % de l'équité
    res = _run([19.9, 19.9, 20.3, 20.3], [20.0, 20.5, 20.4, 20.4], [19.8, 19.95, 20.2, 20.2], [19.9, 20.4, 20.3, 20.3],
               {1: [(0, 20.0, 19.99, 1.0)]}, ema9=[0.0] * 4, liq=False)  # fmt: skip
    assert res.exposure.iloc[1] <= 0.2501


def test_ema9_exit_next_open_and_after_3r_mode():
    o = [19.8, 19.9, 20.0, 19.9, 19.9]
    h = [19.9, 20.5, 20.1, 20.0, 20.0]
    lo = [19.7, 19.8, 19.8, 19.8, 19.8]
    c = [19.8, 20.4, 19.85, 19.9, 19.9]
    e9 = [0.0, 0.0, 20.0, 0.0, 0.0]  # clôture du jour 2 sous l'EMA 9
    always = _run(o, h, lo, c, ORDER, ema9=e9, mode="always")
    assert always.trades.iloc[0]["exit"] == pd.bdate_range("2026-01-05", periods=5)[3]  # vendu à l'ouverture j3
    late = _run(o, h, lo, c, ORDER, ema9=e9, mode="after_3r", liq=False)
    assert late.trades.empty  # 3R jamais atteint : la position reste ouverte


def test_costs_and_tax_reduce_equity():
    o = [19.8, 19.9, 21.0, 22.0, 22.0, 22.0]
    h = [19.9, 20.5, 21.6, 22.7, 22.1, 22.1]
    lo = [19.7, 19.8, 20.8, 21.9, 21.9, 21.9]
    c = [19.8, 20.4, 21.5, 22.6, 22.0, 22.0]
    free = _run(o, h, lo, c, ORDER, ema9=[0.0] * 6, cost=0.0, tax=0.0)
    taxed = _run(o, h, lo, c, ORDER, ema9=[0.0] * 6, cost=25.0, tax=0.30)
    assert free.equity.iloc[-1] > 30_000.0
    assert taxed.fees_eur > 0 and taxed.tax_eur > 0
    assert taxed.equity.iloc[-1] < free.equity.iloc[-1]


def test_fx_conversion_is_applied():
    o = [19.8, 19.9, 19.8, 19.8]
    h = [19.9, 20.5, 19.9, 19.9]
    lo = [19.7, 19.8, 19.4, 19.4]
    c = [19.8, 20.2, 19.6, 19.6]
    res = _run(o, h, lo, c, ORDER, ema9=[0.0] * 4, fx=[1.2] * 4)
    assert res.equity.iloc[-1] == pytest.approx(30_000.0 * (1 - 0.005), rel=2e-3)  # perte de 0,5 % en EUR aussi


def test_vectorized_prefilter_matches_rules_on_synthetic():
    b = synth()
    ok, ar = market.consolidation_prefilter(b)
    for t in range(60, len(b)):
        assert rules.detect_consolidation(b, b.index[t]).ok == bool(ok[t])
    assert ok[-1]  # le cas d'école est bien une consolidation


def test_orders_follow_variant_filters():
    sig = pd.DataFrame(
        {"t": [1, 1, 1], "j": [0, 1, 2], "trigger": [10.0] * 3, "stop": [9.8] * 3, "distance": [0.02, 0.035, 0.02],
         "atr_ratio": [0.5, 0.5, 0.75], "pass80": [True, True, False], "priority": [1.0, 1.0, 1.0]},
    )  # fmt: skip
    base = bench.build_orders(sig, bench.Variant("b"), None, 10)
    assert [o[0] for o in base[2]] == [0, 2]  # distance 3,5 % écartée
    assert [o[0] for o in bench.build_orders(sig, bench.Variant("b", max_dist=0.04), None, 10)[2]] == [0, 1, 2]
    assert [o[0] for o in bench.build_orders(sig, bench.Variant("b", atr_max=0.7), None, 10)[2]] == [0]
    assert [o[0] for o in bench.build_orders(sig, bench.Variant("b", rank80=True), None, 10)[2]] == [0]
    mk = np.zeros(10, dtype=bool)
    assert bench.build_orders(sig, bench.Variant("b", market_filter=True), mk, 10) == {}
    assert bench.build_orders(sig, bench.Variant("b"), None, 2) == {}  # pas d'ordre au-delà de la dernière séance


def test_gate_logic():
    ndx = {"cagr": 0.10, "sharpe": 0.7}
    good = {"cagr": 0.2, "sharpe": 1.0, "max_dd": -0.2, "expectancy_r": 0.4, "n_trades": 200}
    large = {"cagr": 0.1, "expectancy_r": 0.3, "n_trades": 100}
    assert all(r["ok"] for r in bench.gate(good, ndx, 0.02, 0.02, large))
    bad = bench.gate({**good, "max_dd": -0.5}, ndx, 0.02, 0.02, large)
    assert [r["critere"] for r in bad if not r["ok"]] == ["Drawdown maximal >= -35 %"]
    assert not all(r["ok"] for r in bench.gate(good, ndx, 0.2, 0.02, large))


def test_intraday_modes_entry_and_stop_same_day():
    o = [19.8, 19.9, 20.0]
    h = [19.9, 20.5, 20.1]
    lo = [19.7, 19.4, 19.9]  # le jour 1 touche le trigger ET le stop
    bull = [19.8, 20.3, 20.0]  # clôture > ouverture : chemin O-L-H-C, le stop cède avant le déclenchement
    bear = [19.8, 19.6, 20.0]  # clôture < ouverture : O-H-L-C, entrée puis stop
    assert _run(o, h, lo, bull, ORDER, ema9=[0.0] * 3, intraday="worst").trades.iloc[0]["r"] == pytest.approx(-1.0, rel=1e-3)
    assert _run(o, h, lo, bull, ORDER, ema9=[0.0] * 3, intraday="ohlc").trades.empty
    assert _run(o, h, lo, bear, ORDER, ema9=[0.0] * 3, intraday="ohlc").trades.iloc[0]["r"] == pytest.approx(-1.0, rel=1e-3)
    assert _run(o, h, lo, bear, ORDER, ema9=[0.0] * 3, intraday="best").trades.empty
    with pytest.raises(ValueError):
        _run(o, h, lo, bull, ORDER, intraday="optimiste")


def test_intraday_target_before_stop_in_best_mode():
    # jour 2 : la barre touche 3R (21,54) ET le stop ; worst = stop d'abord, best = tiers vendu à 3R puis stop
    o = [19.8, 19.9, 20.4, 20.0]
    h = [19.9, 20.5, 21.6, 20.0]
    lo = [19.7, 19.8, 19.4, 19.9]
    c = [19.8, 20.4, 20.0, 20.0]
    worst = _run(o, h, lo, c, ORDER, ema9=[0.0] * 4, mode="after_3r", intraday="worst").trades.iloc[0]
    best = _run(o, h, lo, c, ORDER, ema9=[0.0] * 4, mode="after_3r", intraday="best").trades.iloc[0]
    assert worst["r"] == pytest.approx(-1.0, rel=1e-3) and worst["stage"] == 0
    assert best["stage"] == 1 and best["r"] == pytest.approx(((21.54 - 20.01) + 2 * (19.5 - 20.01)) / 3 / 0.51, rel=1e-3)
