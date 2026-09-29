"""Banc de vérité (phase 1) : le moteur ne doit jamais regarder le futur et doit facturer coûts et impôt."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import data as bt_data
from src.backtest import engine, metrics, strategies


def _prices(closes, opens=None, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"Open": opens, "Close": closes, "Volume": 1000}, index=idx)


def _target(prices, values):
    return pd.Series(values, index=prices.index, dtype=float)


class TestEngineTiming:
    def test_signal_is_executed_at_next_open_never_same_day(self):
        # Le signal d'achat tombe à la clôture du jour 1 ; le jour 2 ouvre à 200 (gap) : on paie 200, pas 100.
        p = _prices(closes=[100, 100, 210, 210], opens=[100, 100, 200, 210])
        res = engine.run_backtest(p, _target(p, [0, 1, 1, 1]), cost_bps=0)
        buy = res.trades[res.trades["side"] == "BUY"].iloc[0]
        assert buy["date"] == p.index[2] and buy["price"] == pytest.approx(200.0)

    def test_last_target_is_never_executed(self):
        p = _prices([100, 100, 100])
        res = engine.run_backtest(p, _target(p, [0, 0, 1]), cost_bps=0)
        assert res.trades.empty and res.equity.iloc[-1] == pytest.approx(30_000.0)

    def test_future_prices_do_not_change_past_equity(self):
        base = _prices([100, 101, 102, 103, 104, 105, 106])
        alt = base.copy()
        alt.iloc[-1] = [500.0, 500.0, 1000]
        tgt = [0, 1, 1, 1, 1, 1, 1]
        a = engine.run_backtest(base, _target(base, tgt), cost_bps=0).equity
        b = engine.run_backtest(alt, _target(alt, tgt), cost_bps=0).equity
        assert a.iloc[:-1].equals(b.iloc[:-1])


class TestEngineCostsAndTax:
    def test_round_trip_pays_cost_on_both_sides(self):
        p = _prices([100] * 6)
        res = engine.run_backtest(p, _target(p, [1, 1, 0, 0, 0, 0]), cost_bps=100)  # 1 % par côté
        # achat à 101, vente à 99 sur un marché plat : 30000 * (99/101) ≈ 29405,94
        assert res.equity.iloc[-1] == pytest.approx(30_000 * 99 / 101, rel=1e-9)

    def test_buy_and_hold_full_investment_keeps_cash_non_negative(self):
        p = _prices(np.linspace(100, 150, 50))
        res = engine.run_backtest(p, strategies.buy_and_hold(p), cost_bps=25)
        assert (res.exposure <= 1.0 + 1e-9).all() and res.equity.iloc[-1] > 30_000

    def test_tax_on_realized_gain_only(self):
        p = _prices([100, 100, 100, 200, 200, 200])
        res = engine.run_backtest(p, _target(p, [1, 1, 1, 0, 0, 0]), cost_bps=0, tax_rate=0.30)
        # +100 % réalisé sur 30000 = 30000 de gain, impôt 30 % = 9000
        assert res.trades["tax"].sum() == pytest.approx(9_000.0)
        assert res.equity.iloc[-1] == pytest.approx(60_000 - 9_000)

    def test_loss_in_same_year_offsets_earlier_gain(self):
        # gain 1 : 100 -> 200 (+30000), puis perte 200 -> 100 (-50 % sur 60000-9000 investis)
        p = _prices([100, 100, 200, 200, 200, 100, 100, 100])
        tgt = [1, 1, 0, 1, 1, 0, 0, 0]
        res = engine.run_backtest(p, _target(p, tgt), cost_bps=0, tax_rate=0.30)
        # cumul de l'année : gain net <= 0 après la 2e vente => impôt total remboursé à zéro ou proche
        realized_year = res.trades["realized_pnl"].sum()
        expected_tax = 0.30 * max(0.0, realized_year)
        assert res.trades["tax"].sum() == pytest.approx(expected_tax)

    def test_liquidate_at_end_taxes_buy_and_hold(self):
        p = _prices([100, 100, 100, 100, 200])
        held = engine.run_backtest(p, strategies.buy_and_hold(p), cost_bps=0, tax_rate=0.30)
        sold = engine.run_backtest(p, strategies.buy_and_hold(p), cost_bps=0, tax_rate=0.30, liquidate_at_end=True)
        assert held.equity.iloc[-1] == pytest.approx(60_000)
        assert sold.equity.iloc[-1] == pytest.approx(60_000 - 0.30 * 30_000)

    def test_rebalance_days_restores_fixed_exposure(self):
        p = _prices([100] * 5 + [200] * 20)
        res = engine.run_backtest(p, strategies.fixed_exposure(p, 0.5), cost_bps=0, rebalance_days=5)
        assert res.exposure.iloc[-1] == pytest.approx(0.5, abs=0.01)


class TestStrategiesNoLookahead:
    def test_ma_trend_ignores_future_bars(self):
        rng = np.random.default_rng(1)
        closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400))
        p = _prices(closes)
        full = strategies.ma_trend(p, 50)
        cut = strategies.ma_trend(p.iloc[:300], 50)
        assert full.iloc[:300].equals(cut)

    def test_ma_trend_out_of_market_during_warmup(self):
        p = _prices(np.linspace(100, 200, 120))
        t = strategies.ma_trend(p, 100)
        assert (t.iloc[:99] == 0).all() and t.iloc[-1] == 1

    def test_ma_trend_hysteresis_reduces_whipsaw(self):
        rng = np.random.default_rng(3)
        closes = 100 + np.cumsum(rng.normal(0, 1, 500))
        p = _prices(np.abs(closes) + 50)
        flips = lambda s: int((s.diff().abs() > 0).sum())  # noqa: E731
        assert flips(strategies.ma_trend(p, 50, band=0.03)) <= flips(strategies.ma_trend(p, 50, band=0.0))

    def test_momentum_uses_only_past(self):
        p = _prices(np.linspace(100, 300, 300))
        assert strategies.time_series_momentum(p, 252).iloc[-1] == 1.0
        assert strategies.time_series_momentum(p, 252).iloc[:252].eq(0).all()


class TestCurrentExitRules:
    def test_take_profit_truncates_a_trend(self):
        # hausse régulière de 1 %/jour : le TP à +8 % sort au ~8e jour alors que la tendance continue
        p = _prices(100 * 1.01 ** np.arange(60))
        t = strategies.apply_current_exit_rules(p, strategies.buy_and_hold(p))
        assert (t == 0).any(), "le take-profit doit sortir"
        held = engine.run_backtest(p, strategies.buy_and_hold(p), cost_bps=0).equity.iloc[-1]
        cut = engine.run_backtest(p, t, cost_bps=0).equity.iloc[-1]
        assert cut < held

    def test_hard_stop_exits_on_minus_10pct(self):
        p = _prices([100, 100, 95, 89, 89, 89, 89])
        t = strategies.apply_current_exit_rules(p, strategies.buy_and_hold(p), time_stop_days=999)
        assert t.iloc[3] == 0.0 and t.iloc[2] == 1.0

    def test_exit_when_base_signal_turns_off(self):
        p = _prices([100] * 8)
        base = _target(p, [1, 1, 1, 0, 0, 0, 0, 0])
        t = strategies.apply_current_exit_rules(p, base, time_stop_days=999)
        assert list(t.iloc[:4]) == [1.0, 1.0, 1.0, 0.0]

    def test_trailing_triggers_only_when_in_profit(self):
        # entrée à 100, sommet 106, repli de 3,1 % depuis le sommet avec +2,7 % de profit : trailing déclenché
        up = _prices([100, 100, 106, 106, 102.7, 102.7, 102.7], opens=[100, 100, 100, 106, 106, 102.7, 102.7])
        t = strategies.apply_current_exit_rules(up, strategies.buy_and_hold(up), take_profit=0.5, time_stop_days=999)
        assert t.iloc[3] == 1.0 and t.iloc[4] == 0.0
        # même repli de 3 % mais position perdante (-1,1 %) : pas de trailing (c'est le hard-stop qui protège)
        down = _prices([100, 100, 102, 102, 98.9, 98.9, 98.9], opens=[100, 100, 100, 102, 102, 98.9, 98.9])
        t = strategies.apply_current_exit_rules(down, strategies.buy_and_hold(down), take_profit=0.5, time_stop_days=999)
        assert (t.iloc[:5] == 1.0).all()

    def test_time_stop_sells_flat_position_after_15_calendar_days(self):
        p = _prices([100.0] * 30)
        t = strategies.apply_current_exit_rules(p, strategies.buy_and_hold(p))
        first_exit = int(np.argmax((t == 0).to_numpy() & (np.arange(len(t)) > 0)))
        assert (p.index[first_exit] - p.index[1]).days >= 15
        assert (p.index[first_exit - 1] - p.index[1]).days < 15


class TestMetrics:
    def test_max_drawdown_and_cagr(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        eq = pd.Series([100.0, 50.0, 75.0], index=idx)
        assert metrics.max_drawdown(eq) == pytest.approx(-0.5)
        one_year = pd.Series([100.0, 121.0], index=pd.DatetimeIndex(["2020-01-01", "2021-01-01"]))
        assert metrics.cagr(one_year) == pytest.approx(0.21, abs=0.002)

    def test_sharpe_of_constant_returns_is_zero_and_scales_with_drift(self):
        assert metrics.sharpe(np.full(50, 0.001)) == 0.0
        r = np.random.default_rng(0).normal(0.001, 0.01, 1000)
        assert metrics.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))

    def test_compute_metrics_keys_and_trade_stats(self):
        p = _prices([100, 100, 110, 110, 110, 100, 90, 90])
        res = engine.run_backtest(p, _target(p, [1, 1, 0, 0, 1, 1, 0, 0]), cost_bps=0)
        m = metrics.compute_metrics(res)
        assert m["n_trades"] == 2 and m["win_rate"] == 0.5
        assert m["avg_win"] == pytest.approx(0.10) and m["avg_loss"] == pytest.approx(-0.10)
        assert m["max_drawdown"] < 0 and 0 < m["time_invested"] < 1

    def test_bootstrap_detects_a_clear_edge_and_no_edge(self):
        rng = np.random.default_rng(5)
        idx = pd.bdate_range("2020-01-01", periods=1000)
        noise = rng.normal(0, 0.01, 1000)
        better = pd.Series(noise + 0.002, index=idx)
        base = pd.Series(noise, index=idx)
        out = metrics.paired_block_bootstrap_sharpe_diff(better, base, n_boot=300, seed=1)
        assert out["ci_low"] > 0 and out["p_le_0"] < 0.05
        same = metrics.paired_block_bootstrap_sharpe_diff(base, base, n_boot=300, seed=1)
        assert same["diff"] == 0 and same["ci_low"] == 0 == same["ci_high"]

    def test_deflated_sharpe_is_lower_with_more_trials(self):
        r = pd.Series(np.random.default_rng(2).normal(0.0006, 0.01, 1250))
        few = metrics.deflated_sharpe_ratio(r, [0.5, 0.9])
        many = metrics.deflated_sharpe_ratio(r, list(np.linspace(-0.5, 1.2, 60)))
        assert 0.0 <= many <= few <= 1.0


class TestFrozenSeries:
    def _df(self, closes, volumes):
        idx = pd.bdate_range("2022-01-03", periods=len(closes))
        return pd.DataFrame({"Open": closes, "Close": closes, "Volume": volumes}, index=idx, dtype=float)

    def test_live_segment_cuts_after_the_last_long_frozen_run(self):
        closes = [10.0] * 50 + [10.5 + i * 0.1 for i in range(30)]
        volumes = [0] * 50 + [500] * 30
        df = self._df(closes, volumes)
        live = bt_data.live_segment(df, min_run=20)
        assert len(live) == 30 and live["Volume"].gt(0).all()
        cov = bt_data.coverage(df)
        assert cov["frozen_rows"] == 49 and cov["live_rows"] == 30

    def test_short_flat_days_are_not_frozen_runs(self):
        closes = [10.0, 10.0, 10.0, 10.2, 10.3]
        volumes = [0, 0, 0, 100, 100]
        df = self._df(closes, volumes)
        assert len(bt_data.live_segment(df, min_run=20)) == 5

    def test_clean_series_is_returned_untouched(self):
        df = self._df([10 + i for i in range(40)], [100] * 40)
        assert bt_data.live_segment(df).equals(df)


def test_frozen_mask_tolerates_a_missing_volume_column():
    df = pd.DataFrame({"Open": [1.0, 1.0], "Close": [1.0, 1.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    assert not bt_data.frozen_mask(df).any() and len(bt_data.live_segment(df)) == 2
