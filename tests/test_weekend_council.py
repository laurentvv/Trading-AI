"""Unit tests for the weekend council module.

Deterministic, no Ollama/cloud calls — all external I/O is mocked.
Follows the project's unittest + unittest.mock convention.
"""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.council import weekend_council as wc


class TestDfToMarkdown(unittest.TestCase):
    """Dependency-free markdown rendering (replaces pandas.to_markdown/tabulate)."""

    def test_empty_df_returns_empty_string(self):
        self.assertEqual(wc._df_to_markdown(pd.DataFrame()), "")

    def test_renders_header_separator_and_rows(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        md = wc._df_to_markdown(df)
        lines = md.split("\n")
        self.assertEqual(lines[0], "| a | b |")
        self.assertEqual(lines[1], "| --- | --- |")
        self.assertEqual(lines[2], "| 1 | x |")
        self.assertEqual(lines[3], "| 2 | y |")

    def test_pipe_and_newline_escaped(self):
        df = pd.DataFrame({"col": ["a|b", "line1\nline2"]})
        md = wc._df_to_markdown(df)
        self.assertIn("a\\|b", md)
        self.assertIn("line1 line2", md)
        self.assertNotIn("a|b", md.replace("\\|", ""))  # raw pipe must be escaped

    def test_truncation_note_when_exceeding_max_rows(self):
        df = pd.DataFrame({"a": list(range(50))})
        md = wc._df_to_markdown(df, max_rows=5)
        self.assertIn("tronquées", md)


class TestContextBuilding(unittest.TestCase):
    """build_full_context must never raise, even with empty/missing data."""

    @patch.object(wc, "fetch_recent_transactions")
    @patch.object(wc, "fetch_recent_portfolio_state")
    @patch.object(wc, "fetch_recent_model_signals")
    @patch.object(wc, "fetch_recent_logs")
    def test_all_empty(self, mock_logs, mock_signals, mock_portfolio, mock_tx):
        mock_tx.return_value = pd.DataFrame()
        mock_portfolio.return_value = pd.DataFrame()
        mock_signals.return_value = pd.DataFrame()
        mock_logs.return_value = ""
        ctx = wc.build_full_context(days=7)
        self.assertIn("Aucune transaction", ctx)
        self.assertIn("Aucune donnée de portefeuille", ctx)
        self.assertIn("Aucun signal de modèle", ctx)

    @patch.object(wc, "fetch_recent_transactions")
    @patch.object(wc, "fetch_recent_portfolio_state")
    @patch.object(wc, "fetch_recent_model_signals")
    @patch.object(wc, "fetch_recent_logs")
    def test_with_data(self, mock_logs, mock_signals, mock_portfolio, mock_tx):
        mock_tx.return_value = pd.DataFrame({"ticker": ["SXRV.DE"], "type": ["BUY"]})
        mock_portfolio.return_value = pd.DataFrame({"ticker": ["SXRV.DE"], "total_value": [1000.0]})
        mock_signals.return_value = pd.DataFrame(
            {"ticker": ["SXRV.DE"], "model_type": ["hybrid"], "signal": ["BUY"], "confidence": [0.8]}
        )
        mock_logs.return_value = "### Extrait log\n..."
        ctx = wc.build_full_context(days=7)
        self.assertIn("SXRV.DE", ctx)
        self.assertIn("Signaux des Modèles", ctx)
        self.assertIn("Extraits de Logs", ctx)


class TestOllamaChat(unittest.TestCase):
    """_ollama_chat uses /api/chat with structured messages and the assigned model."""

    @patch("requests.post")
    def test_uses_chat_endpoint_and_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Réponse du modèle."}}
        mock_post.return_value = mock_response

        out = wc._ollama_chat("qwen3.5:9b", "system prompt", "user prompt")

        self.assertEqual(out, "Réponse du modèle.")
        sent = mock_post.call_args.kwargs["json"]
        # Must hit the chat endpoint and carry the assigned model.
        self.assertIn("/api/chat", mock_post.call_args.args[0])
        self.assertEqual(sent["model"], "qwen3.5:9b")
        # Structured messages, not a flat prompt.
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertEqual(sent["messages"][1]["role"], "user")

    @patch("requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response
        with self.assertRaises(RuntimeError):
            wc._ollama_chat("missing-model", "sys", "user")

    @patch("requests.post")
    def test_raises_on_empty_response(self, mock_post):
        """A thinking model that burns all tokens on reasoning returns empty."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_post.return_value = mock_response
        with self.assertRaises(RuntimeError):
            wc._ollama_chat("qwen3.5:9b", "sys", "user")

    @patch("requests.post")
    def test_judge_gets_larger_token_budget(self, mock_post):
        """ask_llm forwards num_predict and num_ctx through to the chat payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "verdict"}}
        mock_post.return_value = mock_response
        wc.ask_llm("sys", "user", model=wc.TEXT_LLM_MODEL, num_predict=12000, num_ctx=65536)
        self.assertEqual(mock_post.call_args.kwargs["json"]["options"]["num_predict"], 12000)
        self.assertEqual(mock_post.call_args.kwargs["json"]["options"]["num_ctx"], 65536)


class TestAskLLMRouting(unittest.TestCase):
    """ask_llm routes each persona to its own model, with graceful fallback."""

    @patch.object(wc, "_ollama_chat", return_value="analyse du quant")
    @patch.object(wc, "_model_available", return_value=True)
    def test_routes_to_assigned_model(self, mock_avail, mock_chat):
        out = wc.ask_llm("sys", "user", model="qwen3.5:9b")
        self.assertEqual(out, "analyse du quant")
        # The assigned model is passed through to _ollama_chat.
        mock_chat.assert_called_once()
        self.assertEqual(mock_chat.call_args.args[0], "qwen3.5:9b")

    @patch.object(wc, "_ollama_chat", return_value="fallback response")
    @patch.object(wc, "_model_available", return_value=False)
    def test_falls_back_when_model_missing(self, mock_avail, mock_chat):
        """If a member's model isn't installed, degrade to the canonical default."""
        out = wc.ask_llm("sys", "user", model="hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q6_K")
        self.assertEqual(out, "fallback response")
        # Must have fallen back to TEXT_LLM_MODEL, not the missing model.
        self.assertEqual(mock_chat.call_args.args[0], wc.TEXT_LLM_MODEL)

    @patch.object(wc, "_ollama_chat")
    @patch.object(wc, "_model_available", return_value=True)
    def test_falls_back_when_assigned_model_errors(self, mock_avail, mock_chat):
        """If the assigned model errors at runtime, retry once on the default."""
        mock_chat.side_effect = [RuntimeError("model crashed"), "recovered on default"]
        out = wc.ask_llm("sys", "user", model="lfm2.5-thinking:1.2b-bf16")
        self.assertEqual(out, "recovered on default")
        # Second call must use TEXT_LLM_MODEL.
        self.assertEqual(mock_chat.call_args_list[1].args[0], wc.TEXT_LLM_MODEL)

    @patch.object(wc, "_ollama_chat", side_effect=RuntimeError("everything broken"))
    @patch.object(wc, "_model_available", return_value=True)
    def test_raises_when_default_also_fails(self, mock_avail, mock_chat):
        with self.assertRaises(RuntimeError):
            wc.ask_llm("sys", "user", model=wc.TEXT_LLM_MODEL)


class TestRunCouncil(unittest.TestCase):
    """Orchestration: 3 rounds, graceful degradation on failure."""

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_three_rounds_thirteen_calls(self, mock_ask, mock_ctx):
        # Divergent stances so the dissent-quota does NOT add an extra call.
        n = len(wc.COUNCIL_MEMBERS)
        stances = ["STANCE: BUY (confiance: 70%)", "STANCE: SELL (confiance: 60%)",
                   "STANCE: HOLD (confiance: 40%)", "STANCE: BUY (confiance: 55%)",
                   "STANCE: SELL (confiance: 50%)", "STANCE: HOLD (confiance: 45%)"]
        mock_ask.side_effect = ["reform"] * n + stances[:n] + ["débat"] * n + ["verdict"]
        report = wc.run_council(days=7)

        # 6 members × (Round 0 restate + Round 1 analyse + Round 2 debate)
        # + 1 Judge = 19 LLM calls. (Dissent-quota steelman is conditional.)
        n_members = len(wc.COUNCIL_MEMBERS)
        expected = n_members * 3 + 1
        self.assertEqual(mock_ask.call_count, expected)
        self.assertIn("Verdict du Juge", report)
        self.assertIn("verdict", report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_graceful_degradation_when_all_calls_fail(self, mock_ask, mock_ctx):
        mock_ask.side_effect = RuntimeError("down")
        report = wc.run_council(days=7)
        # Council must still produce a report, not crash.
        self.assertIn("Verdict du Juge", report)
        self.assertIn("ajourné", report)  # judge unavailable notice
        n_members = len(wc.COUNCIL_MEMBERS)
        self.assertEqual(mock_ask.call_count, n_members * 3 + 1)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_all_members_represented_in_report(self, mock_ask, mock_ctx):
        mock_ask.return_value = "réponse"
        report = wc.run_council(days=7)
        for name in wc.COUNCIL_MEMBERS:
            self.assertIn(name, report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_round1_uses_targeted_questions_not_generic(self, mock_ask, mock_ctx):
        """Each member gets its own ROUND1_QUESTIONS prompt, not the generic one."""
        mock_ask.return_value = "STANCE: HOLD (confiance: 50%)"  # parseable stance avoids dissent-quota
        wc.run_council(days=7)
        n = len(wc.COUNCIL_MEMBERS)
        # Round 0 = calls 0..n-1 (restate), Round 1 = calls n..2n-1 (analyse).
        round1_calls = mock_ask.call_args_list[n:2 * n]
        member_names = list(wc.COUNCIL_MEMBERS.keys())
        for i, name in enumerate(member_names):
            user_prompt = round1_calls[i].args[1]  # second positional arg
            targeted = wc.ROUND1_QUESTIONS.get(name, "")
            self.assertIn(targeted[:30], user_prompt,
                          f"{name} should receive its targeted question")
            self.assertNotIn("selon ta perspective", user_prompt,
                             "generic Round 1 prompt should be gone")

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_round2_uses_assigned_contradictor_1v1(self, mock_ask, mock_ctx):
        """Round 2 debates are 1-vs-1: each member faces its assigned opponent."""
        # Divergent stances so the dissent-quota does NOT trigger and shift indices.
        n = len(wc.COUNCIL_MEMBERS)
        stances = ["STANCE: BUY (confiance: 70%)", "STANCE: SELL (confiance: 60%)",
                   "STANCE: HOLD (confiance: 40%)", "STANCE: BUY (confiance: 55%)",
                   "STANCE: SELL (confiance: 50%)", "STANCE: HOLD (confiance: 45%)"]
        mock_ask.side_effect = ["reform"] * n + stances[:n] + ["débat"] * n + ["verdict"]
        wc.run_council(days=7)

        # Round 2 = calls 2n..3n-1 (no dissent-quota call inserted).
        member_names = list(wc.COUNCIL_MEMBERS.keys())
        round2_calls = mock_ask.call_args_list[2 * n:3 * n]
        for i, name in enumerate(member_names):
            opponent = wc.CONTRADICTIONS[name]
            user_prompt = round2_calls[i].args[1]
            self.assertIn(opponent, user_prompt,
                          f"{name} should face its assigned contradictor {opponent}")
            self.assertIn("contradicteur", user_prompt)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_each_member_gets_its_assigned_model(self, mock_ask, mock_ctx):
        """The routing core: every member is called with its own MEMBER_MODELS entry."""
        mock_ask.return_value = "STANCE: HOLD (confiance: 50%)"  # parseable stance avoids dissent-quota
        wc.run_council(days=7)

        member_names = list(wc.COUNCIL_MEMBERS.keys())
        n = len(member_names)
        # Both Round 0 (restate) and Round 1 (analyse) route to the member's own
        # model — check Round 1 here (calls n..2n-1).
        round1_calls = mock_ask.call_args_list[n:2 * n]
        for i, name in enumerate(member_names):
            expected_model = wc.MEMBER_MODELS[name]
            passed_model = round1_calls[i].kwargs.get("model")
            self.assertEqual(passed_model, expected_model,
                             f"{name} must be routed to {expected_model}, got {passed_model}")

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_report_includes_models_used_footer(self, mock_ask, mock_ctx):
        """The report must document which model answered each member (transparency)."""
        mock_ask.return_value = "STANCE: HOLD (confiance: 50%)"  # parseable stance avoids dissent-quota
        report = wc.run_council(days=7)
        self.assertIn("Modèles utilisés", report)
        # Every assigned model must be mentioned in the footer.
        for model in wc.MEMBER_MODELS.values():
            self.assertIn(model, report)
        self.assertIn(wc.JUDGE_MODEL, report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_vote_tally_and_restate_in_report(self, mock_ask, mock_ctx):
        """The report includes a Vote Tally table and the Round 0 reformulations."""
        # Divergent stances so no dissent-quota triggers.
        n = len(wc.COUNCIL_MEMBERS)
        stances = ["STANCE: BUY (confiance: 70%)", "STANCE: SELL (confiance: 60%)",
                   "STANCE: HOLD (confiance: 40%)", "STANCE: BUY (confiance: 55%)",
                   "STANCE: SELL (confiance: 50%)", "STANCE: HOLD (confiance: 45%)"]
        mock_ask.side_effect = ["reform"] * n + stances[:n] + ["débat"] * n + ["verdict"]
        report = wc.run_council(days=7)
        self.assertIn("Décompte des positions", report)
        self.assertIn("Reformulation du problème", report)

    @patch.object(wc, "build_full_context", return_value="CTX")
    @patch.object(wc, "ask_llm")
    def test_dissent_quota_triggers_on_consensus(self, mock_ask, mock_ctx):
        """When ≥2/3 of members converge on a stance, a steelman is forced."""
        # All members HOLD → strong consensus → dissent quota triggers an extra call.
        mock_ask.return_value = "STANCE: HOLD (confiance: 80%)"
        wc.run_council(days=7)
        n = len(wc.COUNCIL_MEMBERS)
        # n members × 3 rounds + 1 Judge + 1 dissent-quota steelman.
        self.assertEqual(mock_ask.call_count, n * 3 + 2)


class TestSaveReport(unittest.TestCase):
    @patch("src.council.weekend_council.Path")
    def test_save_report(self, mock_path_cls):
        mock_file = MagicMock()
        # Path(...) chain: .mkdir(...) returns dir; / "file" returns file
        mock_path_cls.return_value.mkdir.return_value = None
        mock_path_cls.return_value.__truediv__.return_value = mock_file
        mock_file.open.return_value.__enter__.return_value = MagicMock()

        result = wc.save_report("REPORT")
        mock_path_cls.return_value.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(result, mock_file)


if __name__ == "__main__":
    unittest.main()
