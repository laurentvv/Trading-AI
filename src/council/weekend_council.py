"""
Weekend Council — multi-persona LLM retrospective.

Runs as a long-lived, asynchronous analysis (NOT a per-cycle consensus vote).
Three rounds:
  1. Each member independently analyses the week.
  2. Members critique each other's positions.
  3. A Judge synthesises a final verdict with actionable recommendations.

Prose-only: powered by NexusAI-Client across diverse cloud providers (Groq,
Cerebras, Mistral, Cohere, OpenRouter/OrcaRouter/Nvidia, Gemini Free & Pro).
"""

import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

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
from src.llm_client import _run_sync, strip_thinking_debris
from nexusai_client import AIGateway

load_dotenv()

logger = logging.getLogger("WeekendCouncil")


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
    """Fetch the model signals emitted over the window."""
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


PERF_DB_PATH = Path("model_performance.db")
MONITOR_DB_PATH = Path("performance_monitor.db")
JOURNAL_PATH = Path("trading_journal.csv")


def fetch_model_performance(days: int = 7) -> str:
    """Reads model prediction accuracy from model_performance.db."""
    if not PERF_DB_PATH.exists():
        logger.info("model_performance.db absent — skipping model accuracy context.")
        return ""
    try:
        conn = sqlite3.connect(PERF_DB_PATH)
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = (
            "SELECT model_name, signal_predicted, actual_outcome, return_5d "
            "FROM model_predictions WHERE timestamp >= ?"
        )
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        if df.empty:
            return "Aucune prédiction enregistrée sur la période."

        total = len(df)
        with_outcome = df[df["actual_outcome"].notna()]
        if with_outcome.empty:
            return f"{total} prédictions enregistrées (aucun résultat consolidé pour l'instant)."

        correct = (with_outcome["signal_predicted"] == with_outcome["actual_outcome"]).sum()
        accuracy = (correct / len(with_outcome)) * 100

        summary = f"Précision globale des modèles : {accuracy:.1f}% ({correct}/{len(with_outcome)} signaux vérifiés, {total} au total)\n"
        per_model = (
            with_outcome.groupby("model_name")
            .apply(
                lambda g: f"{g['model_name'].iloc[0]}: {(g['signal_predicted'] == g['actual_outcome']).mean()*100:.0f}% ({len(g)})",
                include_groups=False,
            )
            .tolist()
        )
        summary += "Détail par modèle : " + ", ".join(per_model)
        return summary
    except Exception as e:
        logger.error(f"Failed to fetch model performance: {e}")
        return ""


def fetch_portfolio_monitoring() -> str:
    """Reads portfolio health and alerts from performance_monitor.db."""
    if not MONITOR_DB_PATH.exists():
        logger.info("performance_monitor.db absent — skipping portfolio monitoring context.")
        return ""
    try:
        conn = sqlite3.connect(MONITOR_DB_PATH)
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        metrics_query = "SELECT * FROM daily_metrics WHERE date >= ? ORDER BY date DESC LIMIT 5"
        df_metrics = pd.read_sql_query(metrics_query, conn, params=(cutoff_date,))

        alerts_query = "SELECT * FROM system_alerts WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 10"
        df_alerts = pd.read_sql_query(alerts_query, conn, params=(cutoff_date,))
        conn.close()

        parts = []
        if not df_metrics.empty:
            latest = df_metrics.iloc[0]
            parts.append(
                f"État portefeuille au {latest.get('date', '?')}: "
                f"Valeur={latest.get('portfolio_value', 0):.2f}€, "
                f"Drawdown max={latest.get('max_drawdown', 0)*100:.1f}%, "
                f"Sharpe 30j={latest.get('sharpe_ratio_30d', 0):.2f}, "
                f"Win rate={latest.get('win_rate', 0)*100:.0f}%"
            )
        if not df_alerts.empty:
            alerts_summary = [
                f"[{a.get('severity', '?')}] {a.get('alert_type', '?')}: {a.get('message', '')}"
                for _, a in df_alerts.iterrows()
            ]
            parts.append("Alertes récentes :\n  - " + "\n  - ".join(alerts_summary[:5]))

        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Failed to fetch portfolio monitoring: {e}")
        return ""


def fetch_recent_journal_entries(n: int = 15) -> str:
    """Reads the last N lines of trading_journal.csv for trade execution logs."""
    if not JOURNAL_PATH.exists():
        logger.info("trading_journal.csv absent — skipping journal context.")
        return ""
    try:
        df = pd.read_csv(JOURNAL_PATH)
        if df.empty:
            return ""
        recent = df.tail(n)
        cols = [c for c in ["Date", "Ticker", "Signal", "Confidence", "Action_Executed", "Execution_Price", "Reason"] if c in recent.columns]
        return recent[cols].to_string(index=False)
    except Exception as e:
        logger.error(f"Failed to read trading_journal.csv: {e}")
        return ""


def build_full_context(days: int = 7) -> str:
    """Gathers all historical database data into a single structured context string."""
    transactions = fetch_recent_transactions(days)
    portfolio = fetch_recent_portfolio_state()
    signals = fetch_recent_model_signals(days)
    model_perf = fetch_model_performance(days)
    monitoring = fetch_portfolio_monitoring()
    journal = fetch_recent_journal_entries(15)

    tx_str = transactions.to_string(index=False) if not transactions.empty else "Aucune transaction sur la période."
    port_str = portfolio.to_string(index=False) if not portfolio.empty else "Aucun historique de portefeuille disponible."
    sig_str = signals.to_string(index=False) if not signals.empty else "Aucun signal enregistré sur la période."

    context = f"""=== CONTEXTE OPÉRATIONNEL DE LA SEMAINE ({days} derniers jours) ===

1. ÉTAT DU PORTEFEUILLE & MÉTRIQUES CLÉS :
{monitoring if monitoring else port_str}

2. TRANSACTIONS RÉCENTES :
{tx_str}

3. PERFORMANCE & PRÉCISION DES MODÈLES :
{model_perf if model_perf else "Non disponible."}

4. SIGNAUX ÉMIS PAR LES MODÈLES IA :
{sig_str}

5. DERNIÈRES ENTRÉES DU JOURNAL DE TRADING :
{journal if journal else "Non disponible."}
"""
    return context.strip()


async def _async_ask_nexus(
    provider_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    configured = AIGateway.get_configured_providers()
    target_provider = provider_name.lower().strip() if provider_name else None

    if target_provider and target_provider in configured:
        try:
            async with AIGateway(target_provider) as client:
                resp = await client.generate_text(
                    user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False,
                )
                cleaned = strip_thinking_debris(resp.text.strip())
                if cleaned:
                    return cleaned, f"{resp.provider}/{resp.model}"
        except Exception as e:
            logger.warning(f"Target provider '{target_provider}' failed: {e}. Falling back to auto_fallback.")

    async with AIGateway.auto_fallback() as client:
        resp = await client.generate_text(
            user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
        )
        cleaned = strip_thinking_debris(resp.text.strip())
        return cleaned, f"{resp.provider}/{resp.model}"


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    num_predict: int = 4096,
    **kwargs,
) -> str:
    """Queries NexusAI-Client provider or fallback."""
    text, _backend = _run_sync(
        _async_ask_nexus(model or "auto", system_prompt, user_prompt, temperature, num_predict)
    )
    return text


def ask_llm_with_backend(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    num_predict: int = 4096,
) -> tuple[str, str]:
    """Queries NexusAI-Client and returns (response_text, backend_info)."""
    return _run_sync(
        _async_ask_nexus(model or "auto", system_prompt, user_prompt, temperature, num_predict)
    )


_STANCE_RE = re.compile(r"STANCE\s*:\s*(BUY|SELL|HOLD)", re.IGNORECASE)
_CONF_RE = re.compile(r"confiance\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _parse_stance(text: str) -> tuple[str | None, float | None]:
    """Extracts the explicit STANCE: BUY|SELL|HOLD (confiance: XX%) line."""
    sig_match = _STANCE_RE.search(text)
    conf_match = _CONF_RE.search(text)
    signal = sig_match.group(1).upper() if sig_match else None
    confidence = float(conf_match.group(1)) / 100.0 if conf_match else None
    return signal, confidence


def run_council(days: int = 7) -> str:
    """Executes the weekend council process across distinct cloud AI providers."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print(Panel("[bold green]Démarrage du Conseil d'Intelligence Artificielle (NexusAI)[/bold green]"))
    context = build_full_context(days)

    models_used: dict[str, str] = {}

    # ROUND 0: Problem Restate Gate
    console.print("[bold cyan]ROUND 0: Reformulation du problème[/bold cyan]")
    reformulations: dict[str, str] = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        member_model = MEMBER_MODELS.get(name)
        console.print(f"{name} reformule la question [{member_model}]...")
        user_prompt = f"Contexte brut de la semaine:\n{context}\n\n{RESTATE_INSTRUCTION}"
        try:
            resp, backend = ask_llm_with_backend(prompt_data["content"], user_prompt, model=member_model)
            reformulations[name] = resp
            models_used[name] = backend
        except Exception as e:
            logger.error(f"{name} indisponible (Round 0): {e}")
            reformulations[name] = f"*{name} n'a pas pu reformuler ({e}).*"
            models_used[name] = "indisponible"

    # ROUND 1: Independent Analysis
    console.print("[bold cyan]ROUND 1: Analyse Indépendante[/bold cyan]")
    analyses: dict[str, str] = {}
    stances: dict[str, tuple[str | None, float | None]] = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        member_model = MEMBER_MODELS.get(name)
        console.print(f"Interrogation de {name} [{member_model}]...")
        question = ROUND1_QUESTIONS.get(name, "Quelle est ton analyse selon ta perspective ?")
        user_prompt = f"Voici les données de la semaine:\n{context}\n\n{question}{STANCE_SUFFIX}"
        try:
            response, backend = ask_llm_with_backend(prompt_data["content"], user_prompt, model=member_model)
            analyses[name] = response
            stances[name] = _parse_stance(response)
            models_used[name] = backend
        except Exception as e:
            logger.error(f"{name} indisponible: {e}")
            analyses[name] = f"*{name} n'a pas pu analyser la situation ({e}).*"
            stances[name] = (None, None)
            models_used[name] = "indisponible"

    # DISSENT QUOTA: anti-groupthink mechanism
    valid = {n: s[0] for n, s in stances.items() if s[0]}
    if valid:
        from collections import Counter
        most_common_sig, most_common_n = Counter(valid.values()).most_common(1)[0]
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

    # ROUND 2: Directed Debate (1-vs-1)
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
            debates[name] = f"*{name} a quitté le débat ({e}).*"

    # ROUND 3: Synthesis by the Judge
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
        verdict, judge_backend = ask_llm_with_backend(judge_prompt, user_prompt, model=JUDGE_MODEL, num_predict=8192)
        models_used["Le Juge"] = judge_backend
    except Exception as e:
        logger.error(f"Le Juge est indisponible: {e}")
        verdict = "*Le Juge n'a pas pu rendre son verdict. Le conseil est ajourné.*"
        models_used["Le Juge"] = "indisponible"

    # Assemble Final Report
    final_report = "# Rapport du Conseil d'Intelligence Artificielle\n\n"
    final_report += f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

    final_report += "## Décompte des positions\n\n"
    final_report += "| Membre | Stance | Confiance | Modèle / Provider |\n|---|---|---|---|\n"
    for name in COUNCIL_MEMBERS:
        sig, conf = stances.get(name, (None, None))
        sig_str = sig or "—"
        conf_str = f"{conf:.0%}" if conf else "—"
        final_report += f"| {name} | {sig_str} | {conf_str} | `{models_used.get(name, '?')}` |\n"
    final_report += "\n"

    final_report += "## Reformulation du problème (Round 0)\n\n"
    for name, reform in reformulations.items():
        final_report += f"**{name}:** {reform}\n\n"

    final_report += f"## Verdict du Juge\n\n{verdict}\n\n"
    final_report += f"---\n## Annexe : Transcription des Débats\n\n{full_transcript}"
    final_report += "\n\n---\n## Modèles et Fournisseurs Utilisés (NexusAI)\n\n"
    final_report += "| Membre | Provider / Modèle |\n|---|---|\n"
    for name in list(COUNCIL_MEMBERS.keys()) + ["Le Juge"]:
        final_report += f"| {name} | `{models_used.get(name, '?')}` |\n"
    final_report += "\n*Propulsé par NexusAI-Client — multi-fournisseurs cloud sans IA locale.*"

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

    setup_environment("weekend_council.log")

    parser = argparse.ArgumentParser(description="Exécute le Conseil d'IA du week-end.")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours d'historique à analyser.")
    args = parser.parse_args()

    report = run_council(days=args.days)
    save_report(report)
