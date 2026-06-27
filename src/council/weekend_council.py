import logging
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.database import DB_PATH

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

def get_context_summary(days: int = 7) -> str:
    """Compiles recent data into a markdown context string."""
    transactions = fetch_recent_transactions(days)
    portfolio = fetch_recent_portfolio_state()

    context = f"# Contexte du Trading - Derniers {days} jours\n\n"

    context += "## 1. Dernières Transactions\n"
    if not transactions.empty:
        context += transactions.to_markdown(index=False) + "\n\n"
    else:
        context += "Aucune transaction au cours de cette période.\n\n"

    context += "## 2. État Récent du Portefeuille\n"
    if not portfolio.empty:
        context += portfolio.to_markdown(index=False) + "\n\n"
    else:
        context += "Aucune donnée de portefeuille disponible.\n\n"

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
        context += "## 3. Extraits de Logs Récents\n" + logs
    return context

from free_llm_api_keys import FreeLLMClient
from src.council.council_prompts import COUNCIL_MEMBERS, JUDGE_PROMPT

def ask_llm(system_prompt: str, user_prompt: str) -> str:
    """Uses FreeLLMClient to get a response, handling fallback internally if needed."""
    try:
        client = FreeLLMClient(type="texte")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info(f"Querying FreeLLMClient (text mode)...")
        response_text = client.chat(messages, temperature=0.7, max_tokens=1500)
        return response_text
    except Exception as e:
        logger.warning(f"FreeLLMClient failed: {e}. Trying Ollama fallback...")
        # Simple fallback to Ollama using requests
        try:
            import requests
            import json
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "gemma-4-12b-it-GGUF:Q6_K",
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1500}
            }
            resp = requests.post(url, json=payload, timeout=240)
            if resp.status_code == 200:
                return resp.json().get("response", "No response from Ollama")
            return f"Ollama error: {resp.status_code}"
        except Exception as ollama_e:
            logger.error(f"Ollama fallback failed: {ollama_e}")
            return f"Failed to get response: {ollama_e}"

def run_council(days: int = 7) -> str:
    """Executes the weekend council process."""
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    console.print(Panel("[bold green]Démarrage du Conseil d'Intelligence Artificielle[/bold green]"))
    context = build_full_context(days)

    # ROUND 1: Independent Analysis
    console.print("[bold cyan]ROUND 1: Analyse Indépendante[/bold cyan]")
    analyses = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        console.print(f"Interrogation de {name}...")
        user_prompt = f"Voici les données de la semaine:\n{context}\n\nQuelle est ton analyse selon ta perspective ?"
        response = ask_llm(prompt_data["content"], user_prompt)
        analyses[name] = response

    # ROUND 2: Debate
    console.print("[bold cyan]ROUND 2: Le Débat[/bold cyan]")
    debate_transcript = "## Synthèse des Analyses du Round 1\n\n"
    for name, resp in analyses.items():
        debate_transcript += f"### {name}\n{resp}\n\n"

    debates = {}
    for name, prompt_data in COUNCIL_MEMBERS.items():
        console.print(f"{name} prépare sa critique...")
        user_prompt = (
            f"Voici ce que les autres membres ont dit :\n{debate_transcript}\n\n"
            "Critique leurs positions, souligne leurs angles morts, et défends ton point de vue. "
            "Sois direct et incisif, mais constructif."
        )
        response = ask_llm(prompt_data["content"], user_prompt)
        debates[name] = response

    # ROUND 3: Synthesis by the Judge
    console.print("[bold cyan]ROUND 3: Le Verdict du Juge[/bold cyan]")
    full_transcript = debate_transcript + "## Critiques et Débats (Round 2)\n\n"
    for name, resp in debates.items():
        full_transcript += f"### {name}\n{resp}\n\n"

    judge_prompt = JUDGE_PROMPT["content"]
    user_prompt = f"Voici les débats complets de tes conseillers :\n\n{full_transcript}\n\nRends ton verdict final."
    console.print("Le Juge délibère...")
    verdict = ask_llm(judge_prompt, user_prompt)

    # Assemble Final Report
    final_report = f"# Rapport du Conseil d'Intelligence Artificielle\n\n"
    final_report += f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    final_report += f"## Verdict du Juge\n\n{verdict}\n\n"
    final_report += f"---\n## Annexe : Transcription des Débats\n\n{full_transcript}"

    return final_report

def save_report(report_md: str):
    """Saves the council report to the file system."""
    output_dir = Path("docs/council_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = output_dir / f"council_report_{date_str}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"Report saved to {file_path}")
    print(f"\n[+] Rapport complet sauvegardé dans : {file_path}")

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
