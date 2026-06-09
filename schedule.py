import time
import subprocess
import logging
from datetime import datetime, timedelta
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.bootstrap import setup_environment

# Configuration
TICKERS = ["SXRV.DE", "CRUDP.PA"]
INTERVAL_MINUTES = 30
START_HOUR = 8
START_MINUTE = 30
END_HOUR = 18
END_MINUTE = 0
MORNING_BRIEF_HOUR = 1
MORNING_BRIEF_MINUTE = 0

# Setup Logging
setup_environment("scheduler.log")

logger = logging.getLogger("TradingScheduler")
console = Console()


def is_market_open():
    """Vérifie si nous sommes dans la fenêtre de trading (Lun-Ven, 08:30-18:00)"""
    now = datetime.now()
    # 0 = Lundi, 4 = Vendredi
    if now.weekday() > 4:
        return False, "Week-end"

    start_time = now.replace(hour=START_HOUR, minute=START_MINUTE, second=0, microsecond=0)
    end_time = now.replace(hour=END_HOUR, minute=END_MINUTE, second=0, microsecond=0)

    if now < start_time:
        return False, f"Avant marché (Attente {START_HOUR:02d}:{START_MINUTE:02d})"
    if now > end_time:
        return False, f"Après marché (Fermé depuis {END_HOUR:02d}:{END_MINUTE:02d})"

    return True, "Marché Ouvert"


def run_trading_cycle():
    """Lance l'exécution de main.py pour tous les tickers"""
    # Vérification préalable de la santé d'Ollama
    try:
        import requests

        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_ok = resp.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            ollama_ok = False
    except ImportError:
        ollama_ok = True  # Si requests n'est pas dispo, on laisse passer

    if not ollama_ok:
        logger.critical("OLLAMA INDISPONIBLE — cycle de trading ignoré.")
        return

    logger.info(f"🚀 Lancement du cycle de trading pour {TICKERS}")

    try:
        # On lance uv run main.py avec les tickers et le flag t212
        cmd = ["uv", "run", "main.py", "--ticker", *TICKERS, "--t212"]

        # On utilise subprocess pour garder le scheduler propre
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            logger.info("✅ Cycle terminé avec succès")
        else:
            logger.error(f"❌ Erreur lors de l'exécution : Code {result.returncode}")

    except Exception as e:
        logger.error(f"💥 Erreur critique dans le scheduler : {e}")


def run_morning_brief():
    """Lance l'exécution du Morning Brief la nuit/au petit matin"""
    logger.info("🌅 Lancement du Morning Brief")
    try:
        cmd = ["uv", "run", "morning_brief/morning_brief.py"]
        # Redirection des logs vers analyse_morning.log
        with open("analyse_morning.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Lancement {datetime.now().isoformat()} ---\n")
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            
        if result.returncode == 0:
            logger.info("✅ Morning Brief généré avec succès")
        else:
            logger.error(f"❌ Erreur lors du Morning Brief : Code {result.returncode}")
    except Exception as e:
        logger.error(f"💥 Erreur critique lors du Morning Brief : {e}")


def get_dashboard(status_msg, last_run, next_run, morning_brief_status):
    """Génère un joli dashboard pour la console Windows"""
    table = Table(box=None, expand=True)
    table.add_column("Propriété", style="cyan")
    table.add_column("Valeur", style="white")

    table.add_row("Statut", f"[bold]{status_msg}[/bold]")
    table.add_row("Dernier Run", f"{last_run}")
    table.add_row("Prochain Run", f"[bold green]{next_run}[/bold green]")
    table.add_row("Morning Brief", f"{morning_brief_status}")
    table.add_row("Tickers", ", ".join(TICKERS))
    table.add_row("Intervalle", f"{INTERVAL_MINUTES} min")

    return Panel(
        table,
        title="[bold blue]Trading AI - Live Scheduler[/bold blue]",
        subtitle="Appuyez sur Ctrl+C pour arrêter",
        border_style="blue",
    )


def main():
    last_run_time = "Aucun"
    next_run = datetime.now()
    last_morning_brief_date = None

    console.clear()
    console.print(
        Panel(
            "[bold green]Démarrage du Scheduler Trading AI[/bold green]\nMode: Trading 212 DEMO",
            border_style="green",
        )
    )

    try:
        while True:
            open_status, msg = is_market_open()
            now = datetime.now()

            if open_status:
                if now >= next_run:
                    # C'est l'heure de bosser
                    run_trading_cycle()
                    last_run_time = now.strftime("%H:%M:%S")
                    # On calcule le prochain créneau
                    next_run = now + timedelta(minutes=INTERVAL_MINUTES)

                status_display = f"[bold green]ACTIF[/bold green] - {msg}"
            else:
                status_display = f"[bold yellow]VEILLE[/bold yellow] - {msg}"
                
                # Check for Morning Brief outside of trading hours (e.g. 06:00 AM)
                if now.hour == MORNING_BRIEF_HOUR and now.minute >= MORNING_BRIEF_MINUTE:
                    if last_morning_brief_date != now.date():
                        run_morning_brief()
                        last_morning_brief_date = now.date()

            if last_morning_brief_date == now.date():
                mb_status = "[bold green]Terminé aujourd'hui[/bold green]"
            else:
                mb_status = f"[bold yellow]En attente ({MORNING_BRIEF_HOUR:02d}:{MORNING_BRIEF_MINUTE:02d})[/bold yellow]"

            # Affichage Dashboard
            console.clear()
            console.print(
                get_dashboard(
                    status_display,
                    last_run_time,
                    next_run.strftime("%H:%M:%S") if open_status else "À l'ouverture",
                    mb_status
                )
            )

            # Attendre 30 secondes avant de re-checker le scheduler
            time.sleep(30)

    except KeyboardInterrupt:
        console.print("\n[bold red]Scheduler arrêté par l'utilisateur.[/bold red]")
        sys.exit(0)


if __name__ == "__main__":
    main()
