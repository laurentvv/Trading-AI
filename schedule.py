import os
import time
import threading
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
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
COUNCIL_HOUR = 1          # Saturday 01:00 (once per week)
COUNCIL_MINUTE = 0
COUNCIL_DAY = 5           # Saturday (0=Mon ... 5=Sat, 6=Sun)
COUNCIL_DAYS_ANALYZED = 7
COUNCIL_TIMEOUT = 172800    # 48h (tout le week-end) pour laisser le temps aux modèles de réfléchir

# GO-gate 6 (audit 2026-08-19 I1): inter-process instance lock. Concurrent
# scheduler instances were OBSERVED in PROD (3 launches within 61s on 18/08,
# doubled cycles on 19/08) — they double-analyse, fight over the state file
# and can issue concurrent orders.
SCHEDULER_LOCK_FILE = Path("scheduler.lock")
SCHEDULER_LOCK_STALE_SECONDS = 2 * 3600  # a lock untouched for 2h is dead
LOCK_KEEPER_INTERVAL = 30                # refresh cadence (background thread)

# Setup Logging
setup_environment("scheduler.log")

logger = logging.getLogger("TradingScheduler")
console = Console()


def acquire_scheduler_lock(lock_path: Path = SCHEDULER_LOCK_FILE) -> bool:
    """Create the instance lock exclusively. Returns True when this process
    owns it. An existing lock older than SCHEDULER_LOCK_STALE_SECONDS is
    broken (the lock-keeper thread touches the file every 30s, so a live
    scheduler never looks stale — even during a 48h blocking council run)."""
    lock_path = Path(lock_path)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            logger.info(f"🔒 Verrou scheduler acquis ({lock_path}, PID {os.getpid()}).")
            return True
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                continue  # lock vanished between open and stat — retry
            if age > SCHEDULER_LOCK_STALE_SECONDS:
                logger.warning(
                    f"⚠️ Verrou scheduler périmé ({age / 3600:.1f}h sans activité) — "
                    f"casse et reprise."
                )
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            return False


def refresh_scheduler_lock(lock_path: Path = SCHEDULER_LOCK_FILE) -> None:
    try:
        Path(lock_path).touch()
    except OSError:
        pass


def release_scheduler_lock(lock_path: Path = SCHEDULER_LOCK_FILE) -> None:
    try:
        Path(lock_path).unlink()
        logger.info("🔓 Verrou scheduler libéré.")
    except OSError:
        pass


def _start_lock_keeper(stop_event: threading.Event) -> threading.Thread:
    """Background thread keeping the lock mtime fresh even while the main
    loop is blocked in a long subprocess (e.g. the 48h weekend council)."""

    def _keep():
        while not stop_event.wait(LOCK_KEEPER_INTERVAL):
            refresh_scheduler_lock()

    t = threading.Thread(target=_keep, daemon=True, name="scheduler-lock-keeper")
    t.start()
    return t


def _morning_brief_done_today(brief_path: Path = None) -> bool:
    """True when today's brief file already exists (mtime-based). Survives
    scheduler restarts, mirroring the council's disk guard."""
    path = Path(brief_path) if brief_path else Path("morning_brief/output/morning_market_brief.md")
    if not path.exists():
        return False
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date() == datetime.now().date()
    except OSError:
        return False


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
    # Vérification préalable de la disponibilité des fournisseurs IA
    try:
        from src.llm_client import check_ai_health
        ai_ok = check_ai_health()
    except Exception as e:
        logger.warning(f"Impossible de vérifier la santé IA : {e}")
        ai_ok = True

    if not ai_ok:
        logger.critical("AUCUN FOURNISSEUR IA CONFIGURÉ — cycle de trading ignoré.")
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
        output_dir = Path("morning_brief/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tools").mkdir(parents=True, exist_ok=True)

        cmd = ["uv", "run", "morning_brief/morning_brief.py"]
        # Redirection des logs vers analyse_morning.log
        with open("analyse_morning.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Lancement {datetime.now().isoformat()} ---\n")
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            
        if result.returncode == 0:
            logger.info("✅ Morning Brief généré avec succès")
        else:
            logger.error(f"❌ Erreur lors du Morning Brief : Code {result.returncode}")

        # --- FinAcumen Daily Run ---
        logger.info("Lancement de l'analyse profonde FinAcumen (Daily)")
        import json
        
        output_file = output_dir / "morning_market_brief.md"
        finacumen_section = "\n\n## 5. Analyse Qualitative Profonde (FinAcumen)\n"
        
        for ticker in TICKERS:
            logger.info(f"Exécution FinAcumen pour {ticker}...")
            try:
                # On timeout à 1 heure (3600s) pour être hyper large.
                fin_cmd = ["uv", "run", "src/finacumen_main.py", "--ticker", ticker]
                res = subprocess.run(fin_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
                if res.returncode != 0:
                    logger.warning(f"⚠️ FinAcumen s'est terminé avec le code d'erreur {res.returncode} pour {ticker}.")
            except subprocess.TimeoutExpired:
                logger.error(f"⏱ Timeout (3600s) dépassé pour FinAcumen sur {ticker}.")
            except Exception as e:
                logger.error(f"💥 Erreur inattendue lors de l'exécution de FinAcumen pour {ticker}: {e}")
            
            # Récupération du résultat
            state_file = Path("data_cache/finacumen") / f"finacumen_{ticker}.json"
            
            if state_file.exists():
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    signal = data.get("signal", "N/A")
                    conf = data.get("confidence", 0.0)
                    analysis = data.get("analysis", "Aucune analyse disponible.")
                    finacumen_section += f"\n### {ticker}\n- **Signal:** {signal} (Confiance: {conf})\n- **Analyse:** {analysis}\n"
                except Exception as e:
                    finacumen_section += f"\n### {ticker}\n- **Erreur de lecture:** {e}\n"
            else:
                finacumen_section += f"\n### {ticker}\n- **Erreur:** Résultat non généré.\n"

        if not output_file.exists():
            today = datetime.now().strftime("%Y-%m-%d")
            output_file.write_text(f"# Morning Market Brief — {today}\n\n_Note: Morning Brief de base non généré._\n", encoding="utf-8")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(finacumen_section)
        logger.info("✅ Résultats FinAcumen ajoutés au Morning Brief.")

    except Exception as e:
        logger.error(f"💥 Erreur critique lors du Morning Brief : {e}")


def run_weekend_council():
    """Lance le Conseil d'IA le week-end (samedi & dimanche).

    Asynchrone et isolé : le council n'est PAS un vote du consensus temps réel
    (``enhanced_decision_engine``). Il produit un récapitulatif stratégique
    hebdomadaire dans ``docs/council_reports/`` dont le verdict est ensuite
    injecté comme contexte dans le prompt LLM de décision (cf.
    ``get_council_verdict_context`` dans ``llm_client.py``), au même titre que
    le Morning Brief.
    """
    logger.info("🏛️ Lancement du Conseil d'IA (week-end)")
    try:
        cmd = [
            "uv", "run", "python", "-m", "src.council.weekend_council",
            "--days", str(COUNCIL_DAYS_ANALYZED),
        ]
        with open("weekend_council.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Lancement {datetime.now().isoformat()} ---\n")
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=COUNCIL_TIMEOUT)

        if result.returncode == 0:
            logger.info("✅ Conseil d'IA terminé avec succès")
        else:
            logger.error(f"❌ Erreur lors du Conseil d'IA : Code {result.returncode}")

        # Confirm the report was actually produced (defensive — don't trust the
        # exit code alone, the council swallows per-member inference errors).
        from pathlib import Path
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = Path("docs/council_reports") / f"council_report_{date_str}.md"
        if report_path.exists():
            logger.info(f"📄 Rapport du conseil disponible : {report_path}")
        else:
            logger.warning(f"⚠️ Aucun rapport du conseil trouvé à {report_path}")

    except subprocess.TimeoutExpired:
        logger.error(f"⏱ Timeout ({COUNCIL_TIMEOUT}s) dépassé pour le Conseil d'IA.")
    except Exception as e:
        logger.error(f"💥 Erreur critique lors du Conseil d'IA : {e}")


def get_dashboard(status_msg, last_run, next_run, morning_brief_status, council_status):
    """Génère un joli dashboard pour la console Windows"""
    table = Table(box=None, expand=True)
    table.add_column("Propriété", style="cyan")
    table.add_column("Valeur", style="white")

    table.add_row("Statut", f"[bold]{status_msg}[/bold]")
    table.add_row("Dernier Run", f"{last_run}")
    table.add_row("Prochain Run", f"[bold green]{next_run}[/bold green]")
    table.add_row("Morning Brief", f"{morning_brief_status}")
    table.add_row("Conseil (week-end)", f"{council_status}")
    table.add_row("Tickers", ", ".join(TICKERS))
    table.add_row("Intervalle", f"{INTERVAL_MINUTES} min")

    return Panel(
        table,
        title="[bold blue]Trading AI - Live Scheduler[/bold blue]",
        subtitle="Appuyez sur Ctrl+C pour arrêter",
        border_style="blue",
    )


def scheduler_tick(state: dict) -> None:
    """One scheduler pass: trading cycles when the market is open, morning
    brief / weekend council when closed. Raises on unexpected errors — the
    main loop catches them (GO-gate 6, audit 2026-08-19 I2)."""
    open_status, msg = is_market_open()
    now = datetime.now()

    if open_status:
        if now >= state["next_run"]:
            # C'est l'heure de bosser
            run_trading_cycle()
            state["last_run_time"] = now.strftime("%H:%M:%S")
            # On calcule le prochain créneau
            state["next_run"] = now + timedelta(minutes=INTERVAL_MINUTES)

        status_display = f"[bold green]ACTIF[/bold green] - {msg}"
    else:
        status_display = f"[bold yellow]VEILLE[/bold yellow] - {msg}"

        # GO-gate 6 catch-up: run any time from MORNING_BRIEF_HOUR onwards
        # while not already produced today (disk check survives restarts).
        # The old `now.hour == 1` window silently skipped the brief whenever
        # the machine was off between 01:00 and 01:59 — 22 consecutive missed
        # days in the previous PROD run.
        if now.hour >= MORNING_BRIEF_HOUR:
            if state["last_morning_brief_date"] != now.date() and not _morning_brief_done_today():
                run_morning_brief()
                state["last_morning_brief_date"] = now.date()

        # Check for Weekend Council — runs ONCE per week on Saturday
        # at COUNCIL_HOUR (default 01:00). The anti-double-execution guard
        # combines an in-memory flag AND a persistent check: if today's
        # report already exists on disk (e.g. the scheduler crashed
        # mid-council and restarted), skip the run instead of redoing
        # the whole 6-member council.
        if now.weekday() == COUNCIL_DAY:
            if now.hour == COUNCIL_HOUR and now.minute >= COUNCIL_MINUTE:
                if state["last_council_date"] != now.date():
                    report_path = Path("docs/council_reports") / f"council_report_{now.date()}.md"
                    if report_path.exists():
                        logger.info(f"📋 Rapport du council déjà présent ({report_path}) — run sauté.")
                        state["last_council_date"] = now.date()
                    else:
                        run_weekend_council()
                        state["last_council_date"] = now.date()

    if state["last_morning_brief_date"] == now.date():
        mb_status = "[bold green]Terminé aujourd'hui[/bold green]"
    else:
        mb_status = f"[bold yellow]En attente (catch-up dès {MORNING_BRIEF_HOUR:02d}:{MORNING_BRIEF_MINUTE:02d})[/bold yellow]"

    # Council status: only meaningful on Saturday (the weekly run day)
    if now.weekday() == COUNCIL_DAY:
        if state["last_council_date"] == now.date():
            council_status = "[bold green]Terminé aujourd'hui[/bold green]"
        else:
            council_status = f"[bold yellow]En attente ({COUNCIL_HOUR:02d}:{COUNCIL_MINUTE:02d})[/bold yellow]"
    else:
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        council_status = f"[dim]Hors samedi (prochain: {days[COUNCIL_DAY]} {COUNCIL_HOUR:02d}:{COUNCIL_MINUTE:02d})[/dim]"

    # Affichage Dashboard
    console.clear()
    console.print(
        get_dashboard(
            status_display,
            state["last_run_time"],
            state["next_run"].strftime("%H:%M:%S") if open_status else "À l'ouverture",
            mb_status,
            council_status,
        )
    )


def run_loop_iteration(state: dict) -> bool:
    """One pass that NEVER dies on an unexpected exception (GO-gate 6,
    audit 2026-08-19 I2: any non-KeyboardInterrupt error used to kill the
    scheduler silently — no more cycles until a human noticed)."""
    try:
        scheduler_tick(state)
        return True
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.exception(
            "💥 Exception non gérée dans l'itération du scheduler — la boucle CONTINUE (GO-gate 6)."
        )
        return False


def main():
    # GO-gate 6 (audit I1): refuse to run alongside another live instance.
    if not acquire_scheduler_lock():
        logger.critical(
            "❌ Une autre instance du scheduler est ACTIVE (verrou scheduler.lock détenu) — arrêt immédiat."
        )
        console.print(
            Panel(
                "[bold red]Instance dupliquée refusée[/bold red]\n\n"
                "Une autre instance du scheduler détient scheduler.lock.\n"
                "Fermez-la avant d'en relancer une nouvelle.",
                title="Scheduler déjà actif",
                border_style="red",
            )
        )
        sys.exit(1)

    state = {
        "last_run_time": "Aucun",
        "next_run": datetime.now(),
        "last_morning_brief_date": None,
        "last_council_date": None,
    }

    console.clear()
    console.print(
        Panel(
            "[bold green]Démarrage du Scheduler Trading AI[/bold green]\nMode: Trading 212 DEMO",
            border_style="green",
        )
    )

    stop_event = threading.Event()
    _start_lock_keeper(stop_event)

    try:
        while True:
            run_loop_iteration(state)
            # Attendre 30 secondes avant de re-checker le scheduler
            time.sleep(30)

    except KeyboardInterrupt:
        console.print("\n[bold red]Scheduler arrêté par l'utilisateur.[/bold red]")
        sys.exit(0)
    finally:
        stop_event.set()
        release_scheduler_lock()


if __name__ == "__main__":
    main()
