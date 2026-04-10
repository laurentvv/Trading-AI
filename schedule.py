import time
import subprocess
import logging
from datetime import datetime, datetime as dt
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

# Configuration
TICKERS = ["SXRV.DE", "CRUDP.PA"]
INTERVAL_MINUTES = 30
START_HOUR = 8
START_MINUTE = 30
END_HOUR = 18
END_MINUTE = 0

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
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
        return False, f"Avant marché (Attente {START_HOUR}:{START_MINUTE})"
    if now > end_time:
        return False, f"Après marché (Fermé depuis {END_HOUR}:{END_MINUTE})"
    
    return True, "Marché Ouvert"

def run_trading_cycle():
    """Lance l'exécution de main.py pour tous les tickers"""
    logger.info(f"🚀 Lancement du cycle de trading pour {TICKERS}")
    
    try:
        # On lance uv run main.py avec les tickers et le flag t212
        cmd = [
            "uv", "run", "main.py", 
            "--ticker", *TICKERS, 
            "--t212"
        ]
        
        # On utilise subprocess pour garder le scheduler propre
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Cycle terminé avec succès")
        else:
            logger.error(f"❌ Erreur lors de l'exécution : Code {result.returncode}")
            
    except Exception as e:
        logger.error(f"💥 Erreur critique dans le scheduler : {e}")

def get_dashboard(status_msg, last_run, next_run):
    """Génère un joli dashboard pour la console Windows"""
    table = Table(box=None, expand=True)
    table.add_column("Propriété", style="cyan")
    table.add_column("Valeur", style="white")
    
    table.add_row("Statut", f"[bold]{status_msg}[/bold]")
    table.add_row("Dernier Run", f"{last_run}")
    table.add_row("Prochain Run", f"[bold green]{next_run}[/bold green]")
    table.add_row("Tickers", ", ".join(TICKERS))
    table.add_row("Intervalle", f"{INTERVAL_MINUTES} min")
    
    return Panel(
        table, 
        title="[bold blue]Trading AI - Live Scheduler[/bold blue]", 
        subtitle="Appuyez sur Ctrl+C pour arrêter",
        border_style="blue"
    )

def main():
    last_run_time = "Aucun"
    next_run_dt = dt.now()
    
    console.clear()
    console.print(Panel("[bold green]Démarrage du Scheduler Trading AI[/bold green]\nMode: Trading 212 DEMO", border_style="green"))

    try:
        while True:
            open_status, msg = is_market_open()
            now = dt.now()

            if open_status:
                if now >= next_run_dt:
                    # C'est l'heure de bosser
                    run_trading_cycle()
                    last_run_time = now.strftime("%H:%M:%S")
                    next_run_dt = now.replace(second=0, microsecond=0) + \
                                  (dt.now() - now) + \
                                  (dt.now() - dt.now()) # Reset
                    # On calcule le prochain créneau
                    import datetime as dtime
                    next_run_dt = now + dtime.timedelta(minutes=INTERVAL_MINUTES)
                
                status_display = f"[bold green]ACTIF[/bold green] - {msg}"
            else:
                status_display = f"[bold yellow]VEILLE[/bold yellow] - {msg}"
                # Si le marché est fermé, on checke toutes les minutes
                # et on prévoit le prochain run à l'ouverture demain ou lundi
                if now.hour >= END_HOUR:
                    # On attend demain matin 8h30
                    pass 

            # Affichage Dashboard
            console.clear()
            console.print(get_dashboard(
                status_display, 
                last_run_time, 
                next_run_dt.strftime("%H:%M:%S") if open_status else "À l'ouverture"
            ))
            
            # Attendre 30 secondes avant de re-checker le scheduler
            time.sleep(30)

    except KeyboardInterrupt:
        console.print("\n[bold red]Scheduler arrêté par l'utilisateur.[/bold red]")
        sys.exit(0)

if __name__ == "__main__":
    main()
