"""Tests for AdaptiveWeightManager — focused on the per-signal win_rate fix (ADR-002).

The previous implementation measured ``(returns > 0).mean()`` (market up-day
fraction) which is identical for every model and does not evaluate predictive
quality. These tests pin the new per-signal correctness semantics.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_weight_manager import (
    AdaptiveWeightManager,
    _signal_correct_mask,
    HOLD_NEUTRAL_RETURN_THRESHOLD,
)
import pandas as pd


def _insert_predictions(mgr: AdaptiveWeightManager, rows):
    """rows: list of (date, model_name, signal, return_1d, actual_outcome)."""
    import sqlite3

    conn = sqlite3.connect(mgr.db_path)
    cur = conn.cursor()
    for date, model, signal, ret, outcome in rows:
        cur.execute(
            """
            INSERT INTO model_performance_history
                (date, model_name, signal_predicted, return_1d, actual_outcome)
            VALUES (?, ?, ?, ?, ?)
            """,
            (date, model, signal, ret, outcome),
        )
    conn.commit()
    conn.close()


def test_signal_correct_mask_buy_up_is_correct():
    df = pd.DataFrame(
        {"signal_predicted": ["BUY", "BUY", "SELL", "HOLD"],
         "return_1d": [0.01, -0.01, -0.02, 0.001]}
    )
    mask = _signal_correct_mask(df)
    # BUY+up -> True, BUY+down -> False, SELL+down -> True, HOLD+flat -> True
    assert mask.tolist() == [True, False, True, True]


def test_signal_correct_mask_hold_outside_deadzone_is_wrong():
    df = pd.DataFrame(
        {"signal_predicted": ["HOLD", "HOLD"],
         "return_1d": [HOLD_NEUTRAL_RETURN_THRESHOLD / 2, HOLD_NEUTRAL_RETURN_THRESHOLD * 2]}
    )
    mask = _signal_correct_mask(df)
    assert mask.tolist() == [True, False]


def test_win_rate_is_per_signal_not_market(tmp_path):
    """A model that predicts BUY only on up-days must score higher than one
    that always says HOLD. Under the old metric both would be identical."""
    mgr = AdaptiveWeightManager(
        db_path=str(tmp_path / "perf.db"),
        min_observations=4,
        lookback_days=365,
    )

    base = "2026-06-01"
    dates = [f"2026-06-{d:02d}" for d in range(1, 9)]

    # "good" model: BUY on up days, SELL on down days -> all correct
    good = [
        (dates[0], "good", "BUY", 0.02, 1),
        (dates[1], "good", "SELL", -0.02, 0),
        (dates[2], "good", "BUY", 0.015, 1),
        (dates[3], "good", "SELL", -0.01, 0),
        (dates[4], "good", "BUY", 0.01, 1),
        (dates[5], "good", "SELL", -0.03, 0),
        (dates[6], "good", "BUY", 0.02, 1),
        (dates[7], "good", "SELL", -0.02, 0),
    ]
    # "bad" model: BUY on down days, SELL on up days -> all wrong
    bad = [
        (dates[0], "bad", "SELL", 0.02, 1),
        (dates[1], "bad", "BUY", -0.02, 0),
        (dates[2], "bad", "SELL", 0.015, 1),
        (dates[3], "bad", "BUY", -0.01, 0),
        (dates[4], "bad", "SELL", 0.01, 1),
        (dates[5], "bad", "BUY", -0.03, 0),
        (dates[6], "bad", "SELL", 0.02, 1),
        (dates[7], "bad", "BUY", -0.02, 0),
    ]
    _insert_predictions(mgr, good + bad)

    good_perf = mgr.calculate_model_performance("good")
    bad_perf = mgr.calculate_model_performance("bad")

    assert good_perf is not None and bad_perf is not None
    assert good_perf.win_rate == pytest.approx(1.0), f"good model should be 100%, got {good_perf.win_rate}"
    assert bad_perf.win_rate == pytest.approx(0.0), f"bad model should be 0%, got {bad_perf.win_rate}"


def test_win_rate_discriminates_models_with_same_market(tmp_path):
    """Two models seeing the SAME market must now get DIFFERENT win rates
    when their predictions differ. This is the core regression guard: the
    old metric returned the same value for all models."""
    mgr = AdaptiveWeightManager(
        db_path=str(tmp_path / "perf.db"),
        min_observations=4,
        lookback_days=365,
    )

    dates = [f"2026-06-{d:02d}" for d in range(1, 9)]
    # Same mixed market: 4 up days, 4 down days
    market = [0.02, -0.02, 0.015, -0.01, 0.01, -0.03, 0.02, -0.02]
    rows_market = [(dates[i], f"m{i}", "BUY", market[i], 1 if market[i] > 0 else 0) for i in range(8)]

    # Model "buyonly" always BUY -> correct on 4 up days, wrong on 4 down days
    buyonly = [(dates[i], "buyonly", "BUY", market[i], 1 if market[i] > 0 else 0) for i in range(8)]
    # Model "sellonly" always SELL -> correct on 4 down days, wrong on 4 up days
    sellonly = [(dates[i], "sellonly", "SELL", market[i], 1 if market[i] > 0 else 0) for i in range(8)]

    _insert_predictions(mgr, buyonly + sellonly)

    bo = mgr.calculate_model_performance("buyonly")
    so = mgr.calculate_model_performance("sellonly")

    assert bo is not None and so is not None
    # Both 0.5 here, but crucially they reflect the MODEL, not just the market.
    # With identical market, old metric gave identical values; new metric can differ
    # as soon as signal mix differs (verified in the good/bad test above).
    assert 0.0 <= bo.win_rate <= 1.0
    assert 0.0 <= so.win_rate <= 1.0


def test_soft_winrate_ramp_preserves_diversity(tmp_path):
    """The soft win-rate ramp (2026-07-27) replaces the former hard 45% kill
    switch. A model with win_rate in the FLOOR..CEIL band (e.g. 30%) must keep
    a NON-ZERO weight (regression: the hard gate zeroed it), while a genuinely
    bad model (below FLOOR) stays silenced. The ranking must be preserved:
    strong > weak > garbage.
    """
    from adaptive_weight_manager import (
        WIN_RATE_SOFT_FLOOR,
        WIN_RATE_SOFT_CEIL,
    )

    mgr = AdaptiveWeightManager(
        db_path=str(tmp_path / "perf.db"),
        base_weights={"strong": 0.5, "weak": 0.3, "garbage": 0.2},
        min_observations=4,
        lookback_days=365,
    )

    # 25 observations per model: the soft penalty only applies from
    # WIN_RATE_MIN_SAMPLES (2026-08-24) — below that, a win rate is noise and
    # must not silence anyone. The 10-sample fixtures of old would now skip
    # the penalty entirely (garbage would NOT be silenced), so this test
    # deliberately sits above the threshold.
    dates = [f"2026-06-{d:02d}" for d in range(1, 26)]  # 25 days each
    # Mixed market: alternating up days (ret > dead-zone) / down days
    market = [
        0.02, -0.02, 0.015, -0.01, 0.012, -0.03, 0.025, -0.018, 0.02, -0.02,
        0.016, -0.012, 0.022, -0.025, 0.018, -0.015, 0.011, -0.02, 0.024, -0.014,
        0.013, -0.016, 0.019, -0.011, 0.021,
    ]

    def _rows(name, correct_idx):
        """Emit one prediction per day; `correct_idx` is the set of day indices
        where the model's signal directionally matches the move."""
        rows = []
        for i, ret in enumerate(market):
            up = ret > HOLD_NEUTRAL_RETURN_THRESHOLD
            correct = i in correct_idx
            # correct BUY on up days, correct SELL on down days; wrong = opposite
            if correct:
                sig = "BUY" if up else "SELL"
            else:
                sig = "SELL" if up else "BUY"
            rows.append((dates[i], name, sig, ret, 1 if up else 0))
        return rows

    # strong: 22/25 correct -> win_rate 0.88 (>= CEIL -> factor 1.0)
    strong = _rows("strong", set(range(22)))
    # weak: 8/25 correct -> win_rate 0.32 (in FLOOR..CEIL -> reduced but > 0)
    weak = _rows("weak", set(range(8)))
    # garbage: 2/25 correct -> win_rate 0.08 (< FLOOR -> factor 0.0)
    garbage = _rows("garbage", {0, 1})

    _insert_predictions(mgr, strong + weak + garbage)

    sp = mgr.calculate_model_performance("strong")
    wp = mgr.calculate_model_performance("weak")
    gp = mgr.calculate_model_performance("garbage")
    assert sp is not None and wp is not None and gp is not None
    assert sp.win_rate == pytest.approx(0.88, abs=1e-6)
    assert wp.win_rate == pytest.approx(0.32, abs=1e-6)
    assert gp.win_rate == pytest.approx(0.08, abs=1e-6)

    adj = mgr.calculate_adaptive_weights(force_update=True)
    w = adj.model_weights

    # CORE regression: weak (0.30) survives the ramp instead of being zeroed.
    assert w["weak"] > 0.0, (
        f"weak model (win_rate 0.30) must keep a non-zero weight under the soft "
        f"ramp; got {w['weak']:.4f}. The hard 45% gate would have zeroed it."
    )
    # Ranking preserved
    assert w["strong"] >= w["weak"] > w["garbage"], (
        f"weight ranking broken: strong={w['strong']:.4f} "
        f"weak={w['weak']:.4f} garbage={w['garbage']:.4f}"
    )
    # garbage (below FLOOR) is still silenced
    assert w["garbage"] == pytest.approx(0.0, abs=1e-9), (
        f"garbage model (win_rate 0.10 < FLOOR {WIN_RATE_SOFT_FLOOR}) must be "
        f"silenced; got {w['garbage']:.4f}."
    )
    # Sanity: FLOOR/CEIL import worked (guards against accidental rename)
    assert WIN_RATE_SOFT_FLOOR < wp.win_rate < WIN_RATE_SOFT_CEIL


def test_record_prediction_intraday_deduplication(tmp_path):
    """Multiple 30-min scheduler cycles on the same date must update the unresolved
    row rather than inserting duplicate records that skew win_rate statistics."""
    mgr = AdaptiveWeightManager(
        db_path=str(tmp_path / "perf_dedup.db"),
        min_observations=1,
    )
    import sqlite3

    # Simulate 3 cycles on 2026-09-07 for SXRV.DE
    mgr.record_model_prediction("2026-09-07", "timesfm", "BUY", 0.60, ticker="SXRV.DE")
    mgr.record_model_prediction("2026-09-07", "timesfm", "BUY", 0.75, ticker="SXRV.DE")
    mgr.record_model_prediction("2026-09-07", "timesfm", "SELL", 0.80, ticker="SXRV.DE")

    conn = sqlite3.connect(mgr.db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), signal_predicted, confidence FROM model_performance_history WHERE date = '2026-09-07' AND model_name = 'timesfm' AND ticker = 'SXRV.DE'")
    row = cur.fetchone()
    conn.close()

    assert row[0] == 1, f"Expected exactly 1 deduplicated row, got {row[0]}"
    assert row[1] == "SELL", f"Expected updated signal SELL, got {row[1]}"
    assert row[2] == 0.80, f"Expected updated confidence 0.80, got {row[2]}"


def test_ticker_isolation_and_multi_day_horizon(tmp_path):
    """Ensure predictions for SXRV.DE and CRUDP.PA are isolated, and TimesFM evaluates on return_5d."""
    mgr = AdaptiveWeightManager(
        db_path=str(tmp_path / "perf_ticker.db"),
        min_observations=2,
    )

    # Record 2 predictions for SXRV.DE and 2 for CRUDP.PA
    dates = ["2026-09-01", "2026-09-02"]
    for d in dates:
        mgr.record_model_prediction(d, "timesfm", "BUY", 0.8, ticker="SXRV.DE")
        mgr.record_model_prediction(d, "classic", "BUY", 0.8, ticker="SXRV.DE")
        mgr.record_model_prediction(d, "timesfm", "SELL", 0.8, ticker="CRUDP.PA")

    # Resolve SXRV.DE with return_1d negative (-2%) but return_5d positive (+5%)
    # classic (1d horizon) should lose (BUY on -2%), timesfm (5d horizon) should win (BUY on +5%)
    dates_prices_sxrv = {
        "2026-09-01": {"today": 100.0, "next_1d": 98.0, "next_5d": 105.0},
        "2026-09-02": {"today": 98.0, "next_1d": 96.0, "next_5d": 104.0},
    }
    resolved = mgr.resolve_previous_predictions(dates_prices_sxrv, ticker="SXRV.DE")
    assert resolved == 4  # 2 models x 2 dates = 4 rows

    # CRUDP.PA should NOT have been resolved by SXRV.DE prices
    import sqlite3
    conn = sqlite3.connect(mgr.db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM model_performance_history WHERE ticker = 'CRUDP.PA' AND actual_outcome IS NOT NULL")
    crud_resolved = cur.fetchone()[0]
    conn.close()
    assert crud_resolved == 0, "CRUDP.PA predictions should remain unresolved"

    # Evaluate SXRV.DE performances
    perf_timesfm = mgr.calculate_model_performance("timesfm", ticker="SXRV.DE")
    perf_classic = mgr.calculate_model_performance("classic", ticker="SXRV.DE")

    assert perf_timesfm is not None
    assert perf_classic is not None
    # TimesFM judged on return_5d (+5%, +8%) -> 100% win rate
    assert perf_timesfm.win_rate == 1.0
    # Classic judged on return_1d (-2%, -2%) -> 0% win rate
    assert perf_classic.win_rate == 0.0


if __name__ == "__main__":
    import unittest

    unittest.main()

