"""Unit tests for the weekend council module.

Deterministic, no live cloud calls — all external I/O is mocked.
Follows the project's unittest + unittest.mock convention.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.council import weekend_council as wc


class TestContextBuilding(unittest.TestCase):
    """build_full_context must never raise, even with empty/missing data."""

    @patch.object(wc, "fetch_recent_transactions")
    @patch.object(wc, "fetch_recent_portfolio_state")
    @patch.object(wc, "fetch_recent_model_signals")
    @patch.object(wc, "fetch_model_performance")
    @patch.object(wc, "fetch_portfolio_monitoring")
    @patch.object(wc, "fetch_recent_journal_entries")
    def test_all_empty(self, mock_journal, mock_mon, mock_perf, mock_signals, mock_portfolio, mock_tx):
        mock_tx.return_value = pd.DataFrame()
        mock_portfolio.return_value = pd.DataFrame()
        mock_signals.return_value = pd.DataFrame()
        mock_perf.return_value = "Aucune prédiction"
        mock_mon.return_value = "Aucune métrique"
        mock_journal.return_value = "Aucun log journal"
        ctx = wc.build_full_context(days=7)
        self.assertIn("Aucune transaction", ctx)
        self.assertIn("Aucun signal enregistré", ctx)

    @patch.object(wc, "fetch_recent_transactions")
    @patch.object(wc, "fetch_recent_portfolio_state")
    @patch.object(wc, "fetch_recent_model_signals")
    @patch.object(wc, "fetch_model_performance")
    @patch.object(wc, "fetch_portfolio_monitoring")
    @patch.object(wc, "fetch_recent_journal_entries")
    def test_with_data(self, mock_journal, mock_mon, mock_perf, mock_signals, mock_portfolio, mock_tx):
        mock_tx.return_value = pd.DataFrame({"ticker": ["SXRV.DE"], "type": ["BUY"]})
        mock_portfolio.return_value = pd.DataFrame({"ticker": ["SXRV.DE"], "total_value": [1000.0]})
        mock_signals.return_value = pd.DataFrame(
            {"date": ["2026-08-18"], "ticker": ["SXRV.DE"], "model_type": ["hybrid"], "signal": ["BUY"], "confidence": [0.8], "details": [""]}
        )
        mock_perf.return_value = "Précision globale : 80%"
        mock_mon.return_value = "Valeur=1000€"
        mock_journal.return_value = "BUY executed"
        ctx = wc.build_full_context(days=7)
        self.assertIn("SXRV.DE", ctx)
        self.assertIn("SIGNAUX ÉMIS PAR LES MODÈLES IA", ctx)


class TestAskLLMNexus(unittest.TestCase):
    """ask_llm routes calls via NexusAI-Client."""

    @patch("src.council.weekend_council._run_sync")
    def test_ask_llm_returns_text(self, mock_run_sync):
        mock_run_sync.return_value = ("Analyse du conseiller.", "groq/llama-3.3-70b-versatile")
        out = wc.ask_llm("system prompt", "user prompt", model="groq")
        self.assertEqual(out, "Analyse du conseiller.")


class TestRunCouncil(unittest.TestCase):
    """Orchestration: 3 rounds, graceful degradation on failure."""

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm_with_backend")
    @patch.object(wc, "ask_llm")
    def test_three_rounds_execution(self, mock_ask, mock_ask_backend, mock_ctx):
        n = len(wc.COUNCIL_MEMBERS)
        stances = [
            ("STANCE: BUY (confiance: 70%)", "groq/llama3"),
            ("STANCE: SELL (confiance: 60%)", "cerebras/gpt-oss"),
            ("STANCE: HOLD (confiance: 40%)", "mistral/small"),
            ("STANCE: BUY (confiance: 55%)", "cohere/cmd-r"),
            ("STANCE: SELL (confiance: 50%)", "openrouter/qwen"),
            ("STANCE: HOLD (confiance: 45%)", "gemini_free/gemini-2.5-flash"),
        ]
        reforms = [(f"Reformulation {i}", f"backend_{i}") for i in range(n)]
        mock_ask_backend.side_effect = reforms + stances[:n] + [("Verdict final", "gemini_pro/gemini-2.5-pro")]
        mock_ask.return_value = "Débat réponse"

        report = wc.run_council(days=7)

        self.assertIn("Verdict du Juge", report)
        self.assertIn("Verdict final", report)
        self.assertIn("Décompte des positions", report)
        self.assertIn("Reformulation du problème (Round 0)", report)
        self.assertIn("Modèles et Fournisseurs Utilisés (NexusAI)", report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm_with_backend")
    @patch.object(wc, "ask_llm")
    def test_all_members_represented_in_report(self, mock_ask, mock_ask_backend, mock_ctx):
        mock_ask_backend.return_value = ("STANCE: HOLD (confiance: 50%)", "groq/llama3")
        mock_ask.return_value = "Débat"
        report = wc.run_council(days=7)
        for name in wc.COUNCIL_MEMBERS:
            self.assertIn(name, report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm_with_backend")
    @patch.object(wc, "ask_llm")
    def test_round1_uses_targeted_questions(self, mock_ask, mock_ask_backend, mock_ctx):
        mock_ask_backend.return_value = ("STANCE: HOLD (confiance: 50%)", "groq/llama3")
        mock_ask.return_value = "Débat"
        wc.run_council(days=7)
        n = len(wc.COUNCIL_MEMBERS)
        round1_calls = mock_ask_backend.call_args_list[n:2 * n]
        member_names = list(wc.COUNCIL_MEMBERS.keys())
        for i, name in enumerate(member_names):
            user_prompt = round1_calls[i].args[1]
            targeted = wc.ROUND1_QUESTIONS.get(name, "")
            self.assertIn(targeted[:30], user_prompt)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm_with_backend")
    @patch.object(wc, "ask_llm")
    def test_round2_uses_assigned_contradictor_1v1(self, mock_ask, mock_ask_backend, mock_ctx):
        n = len(wc.COUNCIL_MEMBERS)
        stances = [
            ("STANCE: BUY (confiance: 70%)", "backend"),
            ("STANCE: SELL (confiance: 60%)", "backend"),
            ("STANCE: HOLD (confiance: 40%)", "backend"),
            ("STANCE: BUY (confiance: 55%)", "backend"),
            ("STANCE: SELL (confiance: 50%)", "backend"),
            ("STANCE: HOLD (confiance: 45%)", "backend"),
        ]
        reforms = [(f"Reformulation {i}", "backend") for i in range(n)]
        mock_ask_backend.side_effect = reforms + stances[:n] + [("Verdict", "judge_backend")]
        mock_ask.return_value = "Débat response"

        wc.run_council(days=7)

        member_names = list(wc.COUNCIL_MEMBERS.keys())
        round2_calls = mock_ask.call_args_list[:n]
        for i, name in enumerate(member_names):
            opponent = wc.CONTRADICTIONS[name]
            user_prompt = round2_calls[i].args[1]
            self.assertIn(opponent, user_prompt)
            self.assertIn("contradicteur", user_prompt)


class TestSaveReport(unittest.TestCase):
    @patch("src.council.weekend_council.Path")
    def test_save_report(self, mock_path_cls):
        mock_file = MagicMock()
        mock_path_cls.return_value.mkdir.return_value = None
        mock_path_cls.return_value.__truediv__.return_value = mock_file
        mock_file.open.return_value.__enter__.return_value = MagicMock()

        result = wc.save_report("REPORT")
        mock_path_cls.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(result, mock_file)


if __name__ == "__main__":
    unittest.main()
