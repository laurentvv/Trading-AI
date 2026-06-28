"""
Weekend Council — multi-persona LLM retrospective.

Runs as a long-lived, asynchronous analysis (NOT a per-cycle consensus vote).
Three rounds:
  1. Each member independently analyses the week.
  2. Members critique each other's positions.
  3. A Judge synthesises a final verdict with actionable recommendations.

Prose-only: unlike the real-time decision call sites, this module emits free
text, so the dual-layer JSON defence (ADR-001) does NOT apply here.
"""

import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.council.council_prompts import (
    COUNCIL_MEMBERS,
    CONTRADICTIONS,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    MEMBER_MODELS,
    RESTATE_INSTRUCTION,
    ROUND1_QUESTIONS,
    STANCE_SUFFIX,
)
from src.database import DB_PATH
from src.llm_client import OLLAMA_BASE_URL, TEXT_LLM_MODEL, strip_thinking_debris

logger = logging.getLogger("WeekendCouncil")

OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
# Generous timeout: the council runs async over the weekend on CPU, where a
# 12B model can take several minutes per call (no GPU acceleration).
_OLLAMA_TIMEOUT = 3600


def fetch_recent_transactions(days: int = 7) -> pd.DataFrame:
    """Fetch transactions from the last N days."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = "SELECT * FROM transactions WHERE date >= ? ORDER BY date DESC"
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch transactions: {e}")
        return pd.DataFrame()


def fetch_recent_portfolio_state() -> pd.DataFrame:
    """Fetch the latest portfolio states."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM portfolio_history ORDER BY id DESC LIMIT 10"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch portfolio state: {e}")
        return pd.DataFrame()


def fetch_recent_model_signals(days: int = 7) -> pd.DataFrame:
    """Fetch the model signals emitted over the window.

    These are the actual BUY/SELL/HOLD votes the ensemble produced — the most
    relevant material for a trading retrospective.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = (
            "SELECT date, ticker, model_type, signal, confidence, details "
            "FROM model_signals WHERE date >= ? ORDER BY date DESC"
        )
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Failed to fetch model signals: {e}")
        return pd.DataFrame()


# Paths to the PROD performance databases (separate from the trading_history.db
# above). These hold the real model accuracy and live portfolio metrics.
PERF_DB_PATH = Path("model_performance.db")
MONITOR_DB_PATH = Path("performance_monitor.db")
JOURNAL_PATH = Path("trading_journal.csv")


def fetch_model_performance(days: int = 7) -> str:
    """Reads model prediction accuracy from model_performance.db.

    The most valuable retrospective material: did the models actually predict
    correctly? Compares ``signal_predicted`` vs ``actual_outcome`` over the
    window. Handles the common case where recent outcomes are still NULL
    (computed only at J+1/J+5 once the market has moved) by reporting
    predictions-without-outcome separately from resolvable accuracy.
    """
    if not PERF_DB_PATH.exists():
        logger.info("model_performance.db absent — skipping model accuracy context.")
        return ""
    try:
        conn = sqlite3.connect(PERF_DB_PATH)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Accuracy per model (only rows where the outcome is resolvable).
        acc_df = pd.read_sql_query(
            """
            SELECT model_name,
                   COUNT(*) AS predictions,
                   SUM(CASE WHEN signal_predicted = actual_outcome THEN 1 ELSE 0 END) AS correct,
                   ROUND(AVG(confidence), 2) AS avg_confidence
            FROM model_performance_history
            WHERE actual_outcome IS NOT NULL AND date >= ?
            GROUP BY model_name ORDER BY correct * 1.0 / COUNT(*)
            """,
            conn,
            params=(cutoff_date,),
        )

        # Pending predictions (outcome not yet resolved) for the latest day.
        pending_df = pd.read_sql_query(
            """
            SELECT model_name, signal_predicted, confidence
            FROM model_performance_history
            WHERE actual_outcome IS NULL AND date >= ?
            ORDER BY confidence DESC
            """,
            conn,
            params=(cutoff_date,),
        )
        conn.close()

        out = ""
        if not acc_df.empty:
            acc_df["accuracy_%"] = (acc_df["correct"] / acc_df["predictions"] * 100).round(1)
            out += "**Accuracy résolue par modèle (prédiction vs outcome réel) :**\n"
            out += _df_to_markdown(acc_df[["model_name", "correct", "predictions", "accuracy_%", "avg_confidence"]]) + "\n\n"
        if not pending_df.empty:
            out += f"**Prédictions récentes non encore résolues** ({len(pending_df)} signaux, outcome J+1/J+5 en attente) :\n"
            out += _df_to_markdown(pending_df.head(12)) + "\n\n"
        if not out:
            out = "Aucune donnée de performance sur la période.\n\n"
        return out
    except Exception as e:
        logger.error(f"Failed to fetch model performance: {e}")
        return ""


def fetch_performance_metrics() -> str:
    """Reads live portfolio metrics and critical alerts from performance_monitor.db."""
    if not MONITOR_DB_PATH.exists():
        logger.info("performance_monitor.db absent — skipping metrics context.")
        return ""
    try:
        conn = sqlite3.connect(MONITOR_DB_PATH)

        # Latest portfolio value & drawdown per ticker.
        metrics_df = pd.read_sql_query(
            """
            SELECT ticker, portfolio_value, cumulative_return, max_drawdown
            FROM realtime_metrics
            WHERE id IN (SELECT MAX(id) FROM realtime_metrics GROUP BY ticker)
            """,
            conn,
        )

        # Critical + medium alerts (the actionable ones).
        alerts_df = pd.read_sql_query(
            """
            SELECT timestamp, ticker, alert_type, severity, message
            FROM performance_alerts
            WHERE severity IN ('CRITICAL', 'HIGH')
            ORDER BY timestamp DESC LIMIT 15
            """,
            conn,
        )
        conn.close()

        out = ""
        if not metrics_df.empty:
            out += "**État du portefeuille (dernier connu par ticker) :**\n"
            out += _df_to_markdown(metrics_df) + "\n\n"
        if not alerts_df.empty:
            out += f"**Alertes critiques ({len(alerts_df)} récentes) :**\n"
            out += _df_to_markdown(alerts_df) + "\n\n"
        if not out:
            out = "Aucune métrique ni alerte enregistrée.\n\n"
        return out
    except Exception as e:
        logger.error(f"Failed to fetch performance metrics: {e}")
        return ""


def fetch_trading_journal(days: int = 7) -> str:
    """Reads the executed trading journal (CSV) and aggregates signal distribution.

    More reliable than the transactions table (which is often sparse in DEV):
    the journal captures every cycle's final decision and confidence, exposing
    systematic biases (e.g. a model that only ever says BUY).
    """
    if not JOURNAL_PATH.exists():
        logger.info("trading_journal.csv absent — skipping journal context.")
        return ""
    try:
        df = pd.read_csv(JOURNAL_PATH)
        if df.empty or "Timestamp" not in df.columns:
            return ""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df[df["Timestamp"] >= cutoff]
        if df.empty:
            return "Aucune entrée de journal sur la période.\n\n"

        out = ""
        # Signal distribution per ticker — exposes bullish/bearish bias.
        if "Ticker" in df.columns and "FINAL_SIGNAL" in df.columns:
            dist = df.groupby(["Ticker", "FINAL_SIGNAL"]).size().unstack(fill_value=0)
            out += "**Répartition des signaux finaux (biais directionnel) :**\n"
            out += _df_to_markdown(dist.reset_index()) + "\n\n"

        # Average confidence per ticker. The journal stores confidence as a
        # percentage string ("85.0%"), so coerce to numeric first.
        if "Ticker" in df.columns and "Confidence" in df.columns:
            conf_series = pd.to_numeric(df["Confidence"].astype(str).str.rstrip("%"), errors="coerce")
            conf = conf_series.groupby(df["Ticker"]).agg(["mean", "count"]).round(3)
            conf.columns = ["confiance_moyenne_pct", "cycles"]
            out += "**Confiance moyenne par ticker :**\n"
            out += _df_to_markdown(conf.reset_index().rename(columns={"Ticker": "ticker"})) + "\n\n"
        return out
    except Exception as e:
        logger.error(f"Failed to fetch trading journal: {e}")
        return ""


def _df_to_markdown(df: pd.DataFrame, index: bool = False, max_rows: int = 40) -> str:
    """Render a DataFrame as a Markdown table without the ``tabulate`` dependency.

    ``DataFrame.to_markdown`` requires ``tabulate`` which is not (and should not
    need to be) a project dependency just for this rendering. This helper does a
    minimal, dependency-free Markdown table.
    """
    if df.empty:
        return ""
    truncated = df.head(max_rows)
    cols = list(truncated.columns)
    if index:
        cols = ["index", *cols]
        truncated = truncated.reset_index()

    def _cell(v):
        return str(v).replace("\n", " ").replace("|", "\\|")

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(_cell(truncated[c].iloc[i]) for c in cols) + " |"
            for i in range(len(truncated))]
    out = "\n".join([header, separator, *rows])
    if len(df) > max_rows:
        out += f"\n*...({len(df) - max_rows} lignes supplémentaires tronquées)*"
    return out


def get_context_summary(days: int = 7) -> str:
    """Compiles recent data into a markdown context string."""
    transactions = fetch_recent_transactions(days)
    portfolio = fetch_recent_portfolio_state()
    signals = fetch_recent_model_signals(days)

    context = f"# Contexte du Trading - Derniers {days} jours\n\n"

    context += "## 1. Dernières Transactions\n"
    if not transactions.empty:
        context += _df_to_markdown(transactions) + "\n\n"
    else:
        context += "Aucune transaction au cours de cette période.\n\n"

    context += "## 2. État Récent du Portefeuille\n"
    if not portfolio.empty:
        context += _df_to_markdown(portfolio) + "\n\n"
    else:
        context += "Aucune donnée de portefeuille disponible.\n\n"

    context += "## 3. Signaux des Modèles (Ensemble)\n"
    if not signals.empty:
        context += _df_to_markdown(signals) + "\n\n"
    else:
        context += "Aucun signal de modèle enregistré sur la période.\n\n"

    # The next three sections pull from the PROD performance databases.
    # They are the richest material for a retrospective: real model accuracy,
    # critical alerts, and the executed-decision bias. Each is skipped cleanly
    # if its source is absent (try/except inside the fetch).
    perf = fetch_model_performance(days)
    context += "## 4. Performance Réelle des Modèles (Prédiction vs Outcome)\n"
    context += perf if perf else "Données de performance non disponibles.\n\n"

    metrics = fetch_performance_metrics()
    context += "## 5. Métriques Portefeuille & Alertes Critiques\n"
    context += metrics if metrics else "Aucune métrique disponible.\n\n"

    journal = fetch_trading_journal(days)
    context += "## 6. Journal de Trading (Décisions Exécutées)\n"
    context += journal if journal else "Journal non disponible.\n\n"

    return context


def fetch_recent_logs() -> str:
    """Fetch summaries from recent prod logs if available."""
    try:
        audit_path = Path("logs_prod/audit_report.md")
        brief_path = Path("logs_prod/morning_market_brief.md")
        logs = ""

        if audit_path.exists():
            with open(audit_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Limit size to avoid overwhelming context
                logs += f"### Extrait de {audit_path.name}:\n{content[:1500]}...\n\n"

        if brief_path.exists():
            with open(brief_path, "r", encoding="utf-8") as f:
                content = f.read()
                logs += f"### Extrait de {brief_path.name}:\n{content[:1500]}...\n\n"

        return logs
    except Exception as e:
        logger.error(f"Failed to fetch logs: {e}")
        return ""


def build_full_context(days: int = 7) -> str:
    context = get_context_summary(days)
    logs = fetch_recent_logs()
    if logs:
        context += "## 7. Extraits de Logs Récents\n" + logs
    return context


def _ollama_chat(model, system_prompt, user_prompt, temperature=0.7, num_predict=8192, num_ctx=32768):
    """Queries a specific Ollama model in chat mode (/api/chat).

    Uses the structured messages format (system + user) which enforces the
    persona more reliably than the flat /api/generate prompt. Returns the
    cleaned response. Raises RuntimeError if the model is unavailable.

    ``num_predict`` defaults to 8192: thinking models (Qwen, LFM, Gemma-4)
    spend part of their budget on internal reasoning before emitting visible
    output — too tight a cap produces empty responses on long contexts.
    """
    import requests

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=_OLLAMA_TIMEOUT)
    if resp.status_code == 200:
        raw = resp.json().get("message", {}).get("content", "")
        cleaned = strip_thinking_debris(raw)
        if not cleaned:
            raise RuntimeError(
                f"Ollama {model} returned an empty response (likely all thinking "
                f"tokens consumed the budget). Consider a higher num_predict."
            )
        return cleaned
    raise RuntimeError(f"Ollama error ({model}): HTTP {resp.status_code}")


def _model_available(model: str) -> bool:
    """Checks whether a model is installed locally in Ollama."""
    try:
        import requests

        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        installed = {m.get("name", "") for m in resp.json().get("models", [])}
        return model in installed
    except Exception:
        return False


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    num_predict: int = 8192,
    num_ctx: int = 32768,
) -> str:
    """Queries a specific Ollama model per persona (genuine reasoning diversity).

    The core of the council design: each member runs on a DIFFERENT model
    family (Gemma / GLM / Qwen / LFM) so their analyses diverge structurally
    instead of being costume changes on one model.

    Args:
        model: The persona's assigned Ollama model. If ``None`` or unavailable,
            falls back to ``TEXT_LLM_MODEL`` (the canonical default) so the
            council degrades gracefully rather than failing entirely.
        num_predict: Token budget. The Judge gets a larger budget because its
            input (full transcript) is the longest.

    Output is scrubbed of think-channel debris so reports stay readable
    regardless of which backend answered.
    """
    target = model if (model and _model_available(model)) else TEXT_LLM_MODEL
    if model and target != model:
        logger.warning(
            f"Modèle '{model}' non installé — fallback sur {TEXT_LLM_MODEL}. "
            f"La diversité de raisonnement sera réduite pour ce membre."
        )
    try:
        logger.info(f"Querying Ollama model: {target}")
        return _ollama_chat(target, system_prompt, user_prompt, temperature, num_predict, num_ctx)
    except Exception as e:
        logger.error(f"Ollama model {target} failed: {e}")
        if target != TEXT_LLM_MODEL:
            logger.warning(f"Falling back to {TEXT_LLM_MODEL}...")
            return _ollama_chat(TEXT_LLM_MODEL, system_prompt, user_prompt, temperature, num_predict, num_ctx)
        raise RuntimeError(f"Failed to get response from {target}: {e}")


_STANCE_RE = re.compile(r"STANCE\s*:\s*(BUY|SELL|HOLD)", re.IGNORECASE)
_CONF_RE = re.compile(r"confiance\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _parse_stance(text: str) -> tuple[str | None, float | None]:
    """Extracts the explicit STANCE: BUY|SELL|HOLD (confiance: XX%) line.

    Returns (signal, confidence) or (None, None) if the member didn't emit a
    parseable stance. Tolerant to case/spacing — models don't always follow
    the exact format.
    """
    sig_match = _STANCE_RE.search(text)
    conf_match = _CONF_RE.search(text)
    signal = sig_match.group(1).upper() if sig_match else None
    confidence = float(conf_match.group(1)) / 100.0 if conf_match else None
    return signal, confidence


def run_council(days: int = 7) -> str:
    """Executes the weekend council process.

    Each member runs on its OWN assigned Ollama model (``MEMBER_MODELS``) so
    the four analyses come from genuinely different model families — the core
    of the council design. If a member's model is missing, it transparently
    falls back to the canonical default (reduced diversity, logged).
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel("[bold green]Démarrage du Conseil d'Intelligence Artificielle[/bold green]"))
    context = build_full_context(days)

    # Track which model actually answered each member (transparency on
    # diversity — reported in the final document so the reader knows whether
    # they got genuine multi-model reasoning or degraded fallbacks).
    models_used: dict[str, str] = {}

    # ROUND 0: Problem Restate Gate — each member reframes the central question
    # before analysing. If their reformulations diverge, the question itself
    # may be poorly framed. (Problem Restate Gate from the original design.)
    console.print("[bold cyan]ROUND 0: Reformulation du problème[/bold cyan]")
    reformulations: dict[str, str] = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        member_model = MEMBER_MODELS.get(name)
        console.print(f"{name} reformule la question [{member_model}]...")
        user_prompt = f"Contexte brut de la semaine:\n{context}\n\n{RESTATE_INSTRUCTION}"
        try:
            reformulations[name] = ask_llm(prompt_data["content"], user_prompt, model=member_model)
            models_used[name] = member_model or TEXT_LLM_MODEL
        except Exception as e:
            logger.error(f"{name} indisponible (Round 0): {e}")
            reformulations[name] = f"*{name} n'a pas pu reformuler (Erreur d'inférence).*"
            models_used[name] = "indisponible"

    # ROUND 1: Independent Analysis — each member gets a targeted question
    # on its own model, and must end with an explicit STANCE for the tally.
    console.print("[bold cyan]ROUND 1: Analyse Indépendante[/bold cyan]")
    analyses: dict[str, str] = {}
    stances: dict[str, tuple[str | None, float | None]] = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        member_model = MEMBER_MODELS.get(name)
        console.print(f"Interrogation de {name} [{member_model}]...")
        question = ROUND1_QUESTIONS.get(name, "Quelle est ton analyse selon ta perspective ?")
        user_prompt = f"Voici les données de la semaine:\n{context}\n\n{question}{STANCE_SUFFIX}"
        try:
            response = ask_llm(prompt_data["content"], user_prompt, model=member_model)
            analyses[name] = response
            stances[name] = _parse_stance(response)
            models_used[name] = member_model or TEXT_LLM_MODEL
        except Exception as e:
            logger.error(f"{name} indisponible: {e}")
            analyses[name] = f"*{name} n'a pas pu analyser la situation (Erreur d'inférence).*"
            stances[name] = (None, None)
            models_used[name] = "indisponible"

    # DISSENT QUOTA: if a strong majority (≥2/3 of voting members, with a
    # minimum quorum) converged on the same STANCE too early, force the most
    # confident one to steelman the opposite position. Prevents soft consensus
    # (the anti-groupthink mechanism from the original design).
    valid = {n: s[0] for n, s in stances.items() if s[0]}
    if valid:
        from collections import Counter
        most_common_sig, most_common_n = Counter(valid.values()).most_common(1)[0]
        # ≥2/3 majority (rounds up), AND at least 2 voters (a single-vote
        # "majority" is not real convergence worth disrupting).
        dissent_threshold = max(2, -(-2 * len(valid) // 3))
        if most_common_n >= dissent_threshold:
            forced = max(valid, key=lambda n: stances[n][1] or 0)
            console.print(f"[bold yellow]Dissent quota: {forced} forcé à steelmanner l'inverse de {most_common_sig}[/bold yellow]")
            opp = "SELL" if most_common_sig == "BUY" else "BUY" if most_common_sig in ("HOLD", "SELL") else "HOLD"
            member_model = MEMBER_MODELS.get(forced)
            try:
                steel = ask_llm(
                    COUNCIL_MEMBERS[forced]["content"],
                    f"Le conseil converge à {most_common_n}/{len(valid)} sur {most_common_sig}. "
                    f"Donne le MEILLEUR argument possible POUR {opp} (steelman), "
                    f"même si ça contredit ton analyse. Objectivité brute.",
                    model=member_model,
                )
                analyses[forced] += f"\n\n*Steelman forcé ({opp}) :\n{steel}*"
            except Exception as e:
                logger.warning(f"Dissent quota steelman failed for {forced}: {e}")

    # ROUND 2: Directed Debate — each member critiques ONE assigned opponent
    # (1-vs-1) rather than the whole council, on its own model.
    console.print("[bold cyan]ROUND 2: Le Débat (contradicteur assigné)[/bold cyan]")
    round1_transcript = "## Synthèse des Analyses du Round 1\n\n"
    for name, resp in analyses.items():
        sig, conf = stances.get(name, (None, None))
        stance_line = f" **[STANCE: {sig} {f'{conf:.0%}' if conf else ''}]**" if sig else ""
        round1_transcript += f"### {name} [{models_used.get(name, '?')}]{stance_line}\n{resp}\n\n"

    debates: dict[str, str] = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        opponent = CONTRADICTIONS.get(name, name)
        opponent_view = analyses.get(opponent, "*Analyse indisponible.*")
        member_model = MEMBER_MODELS.get(name)
        console.print(f"{name} affronte {opponent} [{member_model}]...")
        user_prompt = (
            f"Voici l'analyse de ton contradicteur désigné, {opponent} :\n\n"
            f"--- Analyse de {opponent} ---\n{opponent_view}\n--- Fin ---\n\n"
            f"Pour mémoire, voici aussi ton analyse du Round 1 :\n{analyses.get(name, '')}\n\n"
            f"Contredis {opponent} sur ses points les plus faibles. Sois précis : "
            f"cite exactement ce avec quoi tu n'es pas d'accord et pourquoi. "
            f"Défends ta propre analyse si {opponent} l'attaque. Sois direct, mais "
            f"argumenté (pas de simple dénigrement)."
        )
        try:
            response = ask_llm(prompt_data["content"], user_prompt, model=member_model)
            debates[name] = response
        except Exception as e:
            logger.error(f"{name} indisponible pour le débat: {e}")
            debates[name] = f"*{name} a quitté le débat (Erreur d'inférence).*"

    # ROUND 3: Synthesis by the Judge (highest-quality model)
    console.print("[bold cyan]ROUND 3: Le Verdict du Juge[/bold cyan]")
    full_transcript = round1_transcript + "## Débats 1-vs-1 (Round 2)\n\n"
    for name, resp in debates.items():
        opponent = CONTRADICTIONS.get(name, "?")
        full_transcript += f"### {name} → contre {opponent}\n{resp}\n\n"

    judge_prompt = JUDGE_PROMPT["content"]
    user_prompt = (
        f"Voici les analyses (Round 1) puis les débats 1-vs-1 (Round 2) de tes conseillers :\n\n"
        f"{full_transcript}\n\nRends ton verdict final en suivant ta structure impérative."
    )
    console.print(f"Le Juge délibère [{JUDGE_MODEL}]...")
    try:
        # The Judge ingests the full transcript (longest input) and must emit a
        # complete structured verdict — give it a larger token budget than members.
        verdict = ask_llm(judge_prompt, user_prompt, model=JUDGE_MODEL, num_predict=12000, num_ctx=65536)
        models_used["Le Juge"] = JUDGE_MODEL
    except Exception as e:
        logger.error(f"Le Juge est indisponible: {e}")
        verdict = "*Le Juge n'a pas pu rendre son verdict (Erreur d'inférence). Le conseil est ajourné.*"
        models_used["Le Juge"] = "indisponible"

    # Assemble Final Report — Vote Tally first (at-a-glance), then verdict,
    # then the full transcript and a transparency footer on models used.
    final_report = "# Rapport du Conseil d'Intelligence Artificielle\n\n"
    final_report += f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

    # Vote Tally table.
    final_report += "## Décompte des positions\n\n"
    final_report += "| Membre | Stance | Confiance | Modèle |\n|---|---|---|---|\n"
    for name in COUNCIL_MEMBERS:
        sig, conf = stances.get(name, (None, None))
        sig_str = sig or "—"
        conf_str = f"{conf:.0%}" if conf else "—"
        final_report += f"| {name} | {sig_str} | {conf_str} | `{models_used.get(name, '?')}` |\n"
    final_report += "\n"

    # Round 0 reformulations (signals if the question was well-framed).
    final_report += "## Reformulation du problème (Round 0)\n\n"
    for name, reform in reformulations.items():
        final_report += f"**{name}:** {reform}\n\n"

    final_report += f"## Verdict du Juge\n\n{verdict}\n\n"
    final_report += f"---\n## Annexe : Transcription des Débats\n\n{full_transcript}"
    final_report += "\n\n---\n## Modèles utilisés (diversité de raisonnement)\n\n"
    final_report += "| Membre | Modèle assigné |\n|---|---|\n"
    for name in list(COUNCIL_MEMBERS.keys()) + ["Le Juge"]:
        final_report += f"| {name} | `{models_used.get(name, '?')}` |\n"
    final_report += (
        "\n*Chaque membre tourne sur une famille de modèle distincte (Gemma / "
        "GLM / Qwen / LFM) pour garantir des raisonnements structurellement "
        "différents, conformément au design original du council.*"
    )

    return final_report


def save_report(report_md: str) -> Path:
    """Saves the council report to the file system. Returns the file path."""
    output_dir = Path("docs/council_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"council_report_{date_str}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Report saved to {file_path}")
    print(f"\n[+] Rapport complet sauvegardé dans : {file_path}")
    return file_path


if __name__ == "__main__":
    import argparse

    from src.bootstrap import setup_environment

    # Initialize basic environment variables if needed
    setup_environment("weekend_council.log")

    parser = argparse.ArgumentParser(description="Exécute le Conseil d'IA du week-end.")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours d'historique à analyser.")
    args = parser.parse_args()

    report = run_council(days=args.days)
    save_report(report)
