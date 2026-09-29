"""Journal d'audit : toutes les voix, le consensus et l'issue RÉELLE de l'exécution T212 (2026-09-29).

Constat : le journal ne montrait que 7 des 11 voix (grebenkov, hmm_model, oil_bench et council pesaient dans
le consensus sans y figurer) et jamais ce qui avait réellement été exécuté ; impossible de reconstituer une
décision a posteriori (condition de la phase 3 « démo de conformité »).
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import main  # noqa: E402
import src.t212_executor as t212  # noqa: E402

OLD_HEADER = [
    "Timestamp", "Ticker", "FINAL_SIGNAL", "Confidence", "Risk_Level", "Risk_Adjusted", "T212_Equity",
    "Model_classic", "Model_llm_text", "Model_llm_visual", "Model_sentiment", "Model_timesfm",
    "Model_tensortrade", "Model_vincent_ganne",
]


def _decision(final="BUY", **votes):
    voices = [
        SimpleNamespace(model_name=name, signal=sig, confidence=conf)
        for name, (sig, conf) in votes.items()
    ]
    return SimpleNamespace(
        final_signal=final, individual_decisions=voices, consensus_score=0.2345, disagreement_factor=0.5
    )


def _write(decision, results=None, signal="BUY"):
    with patch("main.load_t212_state", return_value={"equity": 1002.5}), \
         patch("t212_executor.get_t212_ticker", return_value="SXRVd_EQ"):
        main._write_trading_journal("SXRV.DE", decision, 0.2, "LOW", signal, True, results)


def _read():
    return pd.read_csv("trading_journal.csv")


class TestJournalColumns:
    def test_new_journal_has_every_voice_score_and_action(self):
        votes = dict(
            classic=("HOLD", 0.52), grebenkov=("BUY", 0.7), hmm_model=("SELL", 0.4),
            oil_bench=("BUY", 0.3), council=("HOLD", 0.6),
        )
        _write(_decision(**votes), results={"t212_action": "BUY:filled"})

        df = _read()
        assert list(df.columns) == main.JOURNAL_HEADER
        row = df.iloc[0]
        assert row["Model_grebenkov"] == "BUY(0.70)"
        assert row["Model_hmm_model"] == "SELL(0.40)"
        assert row["Model_oil_bench"] == "BUY(0.30)"
        assert row["Model_council"] == "HOLD(0.60)"
        assert pd.isna(row["Model_sentiment"])  # "N/A" : pandas le lit comme NaN (valeur historique du journal)
        assert row["Consensus_Score"] == 0.2345
        assert row["T212_Action"] == "BUY:filled"
        assert row["T212_Equity"] == "1002.50 €"

    def test_missing_results_leaves_action_empty(self):
        _write(_decision(classic=("HOLD", 0.5)))
        assert pd.isna(_read().iloc[0]["T212_Action"])


class TestJournalMigration:
    def _old_journal(self):
        with open("trading_journal.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(OLD_HEADER)
            w.writerow(["2026-09-29 09:00:00", "SXRV.DE", "BUY", "32.16%", "LOW", "BUY", "1004.29 €",
                        "HOLD(0.52)", "HOLD(0.95)", "BUY(0.78)", "HOLD(0.50)", "SELL(0.50)", "HOLD(0.36)", "N/A"])
            w.writerow(["2026-09-29 09:30:00", "CRUDP.PA", "SELL", "16.68%", "HIGH", "HOLD", "995.11 €",
                        "SELL(0.63)", "SELL(0.85)", "HOLD(0.50)", "HOLD(0.50)", "SELL(0.83)", "SELL(0.35)", "N/A"])

    def test_old_journal_is_extended_without_shifting_existing_columns(self):
        self._old_journal()
        _write(_decision(grebenkov=("BUY", 0.7)), results={"t212_action": "HOLD"})

        df = _read()
        assert list(df.columns)[: len(OLD_HEADER)] == OLD_HEADER  # anciennes colonnes intactes et à leur place
        assert len(df) == 3
        # anciennes lignes conservées, nouvelles colonnes vides
        assert df.iloc[0]["Model_timesfm"] == "SELL(0.50)"
        assert pd.isna(df.iloc[0]["T212_Action"]) and pd.isna(df.iloc[0]["Model_grebenkov"])
        # nouvelle ligne complète
        assert df.iloc[2]["Model_grebenkov"] == "BUY(0.70)"
        assert df.iloc[2]["T212_Action"] == "HOLD"

    def test_migration_is_idempotent(self):
        self._old_journal()
        _write(_decision(classic=("HOLD", 0.5)))
        first = Path("trading_journal.csv").read_text(encoding="utf-8")
        header = main._ensure_journal_header(Path("trading_journal.csv"), main.JOURNAL_HEADER)
        assert header == main.JOURNAL_HEADER
        assert Path("trading_journal.csv").read_text(encoding="utf-8") == first
        assert not Path("trading_journal.csv.tmp").exists()

    def test_unknown_extra_columns_of_an_old_journal_are_kept(self):
        with open("trading_journal.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(OLD_HEADER + ["Custom_Note"])
            w.writerow(["2026-09-29 09:00:00", "SXRV.DE", "BUY", "1%", "LOW", "BUY", "1000.00 €"]
                       + ["HOLD(0.5)"] * 7 + ["a note"])
        _write(_decision(classic=("HOLD", 0.5)))
        df = _read()
        assert "Custom_Note" in df.columns
        assert df.iloc[0]["Custom_Note"] == "a note"


def _run_trade(signal, *, current_pos=None, positions_ok=True, exit_result=(False, None), state=None,
               min_holding=False, after=None):
    """Exécute execute_t212_trade avec tous les appels réseau/état remplacés."""
    state = state if state is not None else {"active_position": None}
    portfolio = {"positions": [current_pos] if current_pos else [], "cash": 1000.0, "cash_ok": True,
                 "positions_ok": positions_ok}

    def fake_sell(*a, **k):
        state["active_position"] = None if after == "sold" else state.get("active_position")

    def fake_buy(*a, **k):
        if after == "bought":
            state["active_position"] = {"ticker": "SXRVd_EQ"}

    with patch.object(t212, "get_auth_header", return_value={}), \
         patch.object(t212, "load_portfolio_state", return_value=state), \
         patch.object(t212, "_get_portfolio_info", return_value=portfolio), \
         patch.object(t212, "_validate_and_recalibrate_entry_price", side_effect=lambda s, *a, **k: s), \
         patch.object(t212, "manage_open_position", return_value=exit_result), \
         patch.object(t212, "_evaluate_min_holding", return_value=min_holding), \
         patch.object(t212, "_execute_buy_order", side_effect=fake_buy), \
         patch.object(t212, "_execute_sell_order", side_effect=fake_sell):
        return t212.execute_t212_trade(signal, 0.5, ticker="SXRV.DE")


POS = {"instrument": {"ticker": "SXRVd_EQ"}, "quantity": 1.0}


class TestExecuteOutcome:
    def test_unknown_broker_state_aborts(self):
        assert _run_trade("BUY", positions_ok=False) == "ABORT:broker-state-unknown"

    def test_hold_without_exit(self):
        assert _run_trade("HOLD") == "HOLD"

    def test_exit_strategy_filled_and_not_filled(self):
        sold = {"active_position": None}
        assert _run_trade("HOLD", current_pos=POS, exit_result=(True, "trailing-stop"), state=sold) == (
            "EXIT:trailing-stop:filled"
        )
        held = {"active_position": {"ticker": "SXRVd_EQ"}}
        assert _run_trade("HOLD", current_pos=POS, exit_result=(True, "hard-stop-loss"), state=held) == (
            "EXIT:hard-stop-loss:not-filled"
        )

    def test_buy_filled_not_filled_and_already_long(self):
        assert _run_trade("BUY", after="bought") == "BUY:filled"
        assert _run_trade("BUY") == "BUY:not-filled"
        assert _run_trade("BUY", current_pos=POS, state={"active_position": {"ticker": "X"}}) == (
            "BUY:skipped-already-long"
        )

    def test_sell_variants(self):
        assert _run_trade("SELL") == "SELL:no-position"
        held = {"active_position": {"ticker": "SXRVd_EQ"}}
        assert _run_trade("SELL", current_pos=POS, state=dict(held), min_holding=True) == "SELL:blocked-min-holding"
        assert _run_trade("SELL", current_pos=POS, state=dict(held), after="sold") == "SELL:filled"
        # garde anti-perte / échec : la position reste ouverte
        assert _run_trade("SELL", current_pos=POS, state=dict(held)) == "SELL:not-executed"


class TestMainRecordsTheOutcome:
    def test_outcome_is_stored_in_results_for_the_journal(self):
        from src.advanced_risk_manager import AdvancedRiskManager, RiskLevel, RiskMetrics

        system = MagicMock()
        system.risk_manager = AdvancedRiskManager()
        decision = MagicMock()
        decision.final_signal = "BUY"
        decision.risk_adjusted_signal = "BUY"
        decision.final_confidence = 0.6
        risk = RiskMetrics(
            volatility_risk=0.2, drawdown_risk=0.1, correlation_risk=0.2, liquidity_risk=0.1,
            overall_risk_score=0.2, risk_level=RiskLevel.LOW,
        )
        results = {"risk_adjusted_signal": "BUY", "market_data": {"price_series": pd.Series([100.0, 101.0, 102.0])}}
        with patch("main.load_t212_state", return_value={"active_position": None}), \
             patch("main.execute_t212_trade", return_value="BUY:filled"):
            main._execute_t212_orders("SXRV.DE", system, decision, risk, results, None, MagicMock())
        assert results["t212_action"] == "BUY:filled"

    def test_flat_without_trade_signal_is_recorded(self):
        from src.advanced_risk_manager import AdvancedRiskManager, RiskLevel, RiskMetrics

        system = MagicMock()
        system.risk_manager = AdvancedRiskManager()
        decision = MagicMock()
        decision.final_signal = "HOLD"
        decision.risk_adjusted_signal = "HOLD"
        decision.final_confidence = 0.3
        risk = RiskMetrics(
            volatility_risk=0.2, drawdown_risk=0.1, correlation_risk=0.2, liquidity_risk=0.1,
            overall_risk_score=0.2, risk_level=RiskLevel.LOW,
        )
        results = {"risk_adjusted_signal": "HOLD", "market_data": {"price_series": pd.Series([100.0, 101.0])}}
        with patch("main.load_t212_state", return_value={"active_position": None}), \
             patch("main.execute_t212_trade") as mock_exec:
            main._execute_t212_orders("SXRV.DE", system, decision, risk, results, None, MagicMock())
        mock_exec.assert_not_called()
        assert results["t212_action"] == "NONE:flat-no-trade-signal"
