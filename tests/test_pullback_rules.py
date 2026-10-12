"""Règles pullback (plan §1, contrat 6-13) : cas synthétiques déterministes."""


import numpy as np
import pandas as pd
import pytest

from src.pullback import rules

def synth(extra=0):
    n_up=150
    up = 20*np.exp(np.linspace(0,np.log(2.2),n_up))          # tendance haussière
    peak = [up[-1]*f for f in (1.03,1.06,1.09,1.11)]         # sommet
    # consolidation : oscillations qui se resserrent, creux ascendants
    base = peak[-1]
    cons = [base*0.98, base*0.95, base*0.93, base*0.945, base*0.96, base*0.955, base*0.94, base*0.935,
            base*0.945, base*0.955, base*0.95, base*0.94, base*0.945, base*0.95, base*0.955, base*0.957,
            base*0.953, base*0.95, base*0.952, base*0.955]
    close = np.array(list(up)+peak+cons)
    idx = pd.bdate_range(end="2026-10-09", periods=len(close))
    prev = np.r_[close[0], close[:-1]]
    amp = np.r_[np.full(n_up+4, 0.012), np.linspace(0.02,0.006,len(cons))]
    o = prev
    h = np.maximum(o,close)*(1+amp/2)
    l = np.minimum(o,close)*(1-amp/2)
    df = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":close,"Volume":2_000_000.0}, index=idx)
    return df


def test_setup_found_on_clean_pullback():
    b = synth()
    s = rules.find_setup(b, b.index[-1])
    assert s is not None
    assert s.trigger == pytest.approx(b["High"].iloc[-1]) and s.stop == pytest.approx(b["Low"].iloc[-1])
    assert s.distance <= 0.03 and s.confluence >= 2  # EMA 9/21 + AVWAP


def test_downtrend_is_rejected():
    n = 200
    close = np.linspace(80, 40, n)
    idx = pd.bdate_range(end="2026-10-09", periods=n)
    b = pd.DataFrame(
        {"Open": close * 1.005, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 2e6}, index=idx
    )
    c = rules.detect_consolidation(b, idx[-1])
    assert not c.ok and rules.find_setup(b, idx[-1]) is None


def test_wide_stop_is_ignored():
    b = synth()
    b.iloc[-1, b.columns.get_loc("Low")] = b["Low"].iloc[-1] * 0.93  # amplitude du jour > 3 %
    assert rules.find_setup(b, b.index[-1]) is None


def test_no_lookahead_future_bars_do_not_change_decision():
    b = synth()
    asof = b.index[-6]
    before = rules.find_setup(b, asof)
    b2 = b.copy()
    b2.iloc[-5:] = b2.iloc[-5:] * 3.0  # futur bouleversé
    assert rules.find_setup(b2, asof) == before
    assert rules.detect_consolidation(b2, asof) == rules.detect_consolidation(b, asof)
    assert rules.below_ema9_exit(b2, asof) == rules.below_ema9_exit(b, asof)


def test_avwap_numeric():
    idx = pd.bdate_range("2026-10-01", periods=3)
    b = pd.DataFrame(
        {"High": [12, 13, 14], "Low": [8, 9, 10], "Close": [10, 11, 12], "Volume": [100, 200, 100]}, index=idx
    )
    # prix typiques 10, 11, 12 ; ancre = séance 1 : (11*200 + 12*100) / 300
    assert rules.avwap(b, 1).iloc[-1] == pytest.approx((11 * 200 + 12 * 100) / 300)


def test_momentum_screen_thresholds_and_rank():
    idx = pd.bdate_range(end="2026-10-09", periods=200)
    flat = pd.Series(10.0, index=idx)
    strong = pd.Series(np.linspace(10, 20, 200), index=idx)  # +100 % : fort
    mild = pd.Series(np.linspace(10, 11, 200), index=idx)  # < 30 %
    closes = {"FLAT1": flat, "FLAT2": flat, "FLAT3": flat, "MILD": mild, "STRONG": strong}
    res = rules.momentum_screen(closes, idx[-1])
    assert "STRONG" in res and "MILD" not in res and "FLAT1" not in res
    short = {"NEW": pd.Series(np.linspace(1, 2, 30), index=idx[-30:]), "STRONG": strong}
    assert rules.momentum_screen(short, idx[-1])["STRONG"]["horizons"]  # historique court = NaN, pas d'erreur


def test_position_size_risk_and_caps():
    # 0,5 % de 10 000 = 50 $ ; 0,50 $/action -> 100 actions (2 000 $, 20 % <= 25 %)
    assert rules.position_size(10_000, 20.0, 19.5) == pytest.approx(100.0)
    # stop très serré : 50 $ / 0,10 = 500 actions, mais le plafond de 25 % par ligne limite à 125
    assert rules.position_size(10_000, 20.0, 19.9) == pytest.approx(125.0)
    # risque ouvert total : il ne reste que 20 $ de budget de risque -> 40 actions
    assert rules.position_size(10_000, 20.0, 19.5, open_risk=280.0) == pytest.approx(40.0)
    # exposition : il ne reste que 600 $ -> 30 actions
    assert rules.position_size(10_000, 20.0, 19.5, open_exposure=9_400.0) == pytest.approx(30.0)
    assert rules.position_size(10_000, 20.0, 19.5, open_risk=300.0) == 0.0
    assert rules.position_size(10_000, 20.0, 20.0) == 0.0


def test_exit_targets_3r_5r():
    t = rules.exit_targets(20.0, 19.5)
    assert [round(p, 2) for p, _ in t] == [21.5, 22.5]
    assert sum(f for _, f in t) == pytest.approx(2 / 3)  # le dernier tiers suit l'EMA 9


def test_below_ema9_exit():
    b = synth()
    assert not rules.below_ema9_exit(b, b.index[-1])
    b.iloc[-1, b.columns.get_loc("Close")] = b["Close"].iloc[-1] * 0.9
    assert rules.below_ema9_exit(b, b.index[-1])
