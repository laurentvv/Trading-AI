"""
Trading AI - Unified Entry Point
This script performs a full analysis: data fetching, model training, and hybrid decision making.
Usage: python main.py --ticker QQQ
"""

import logging
import sys
import argparse
import csv
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import system modules
from enhanced_trading_example import EnhancedTradingSystem
from t212_executor import execute_t212_trade, load_portfolio_state as load_t212_state
from database import get_latest_portfolio_state, get_latest_transaction
from llm_client import check_ollama_health
from bootstrap import setup_environment

# Load environment
load_dotenv()

# Setup logging
setup_environment("trading.log")
logger = logging.getLogger("TradingAI")

# Cycle timeout: 15 minutes per ticker. Prevents infinite hangs on LLM calls.
CYCLE_TIMEOUT_SECONDS = 15 * 60

# Per-ticker locks preventing concurrent execution of run_trading_analysis
# on the same ticker (defense against orphan threads from a previous cycle
# timeout that may still be running). Keyed by ticker symbol.
_TICKER_LOCKS: dict[str, threading.Lock] = {}
_TICKER_LOCKS_GUARD = threading.Lock()


def _get_ticker_lock(ticker: str) -> threading.Lock:
    """Returns (or creates) a per-ticker lock for serializing T212 trades."""
    with _TICKER_LOCKS_GUARD:
        lock = _TICKER_LOCKS.get(ticker)
        if lock is None:
            lock = threading.Lock()
            _TICKER_LOCKS[ticker] = lock
        return lock


# Cancel events per ticker: set by the cycle timeout handler so the orphan
# worker thread can bail out before placing a real T212 order.
_TICKER_CANCEL_EVENTS: dict[str, threading.Event] = {}


def _get_cancel_event(ticker: str) -> threading.Event:
    """Returns (or creates) a per-ticker cancel event."""
    ev = _TICKER_CANCEL_EVENTS.get(ticker)
    if ev is None:
        ev = threading.Event()
        _TICKER_CANCEL_EVENTS[ticker] = ev
    return ev


def check_setup() -> bool:
    """Vérifie si TimesFM 2.5 est correctement installé et patché"""
    vendor_path = Path(__file__).parent / "vendor" / "timesfm"
    if not vendor_path.exists():
        console = Console()
        console.print(
            Panel(
                "[bold red]ERREUR : TimesFM 2.5 n'est pas installé.[/bold red]\n\n"
                "[*] Veuillez lancer la commande suivante pour tout configurer automatiquement :\n"
                "    [bold cyan]uv run setup[/bold cyan]",
                title="Setup Manquant",
                border_style="red",
            )
        )
        return False
    return True


def run_trading_analysis(
    ticker: str,
    is_simulation: bool = False,
    is_t212: bool = False,
    cancel_event: threading.Event | None = None,
):
    if not check_setup():
        return

    console = Console()

    # Vérification santé Ollama avant chaque cycle
    ollama_ok = check_ollama_health()
    if not ollama_ok:
        logger.critical(f"OLLAMA INDISPONIBLE pour {ticker} — le cycle de trading est ignoré.")
        console.print(
            Panel(
                f"[bold red]OLLAMA INDISPONIBLE (localhost:11434)[/bold red]\n\n"
                f"Le cycle pour {ticker} est ignoré. Les modèles LLM sont indispensables.\n"
                f"Vérifiez que le service Ollama est démarré.",
                title="Ollama Healthcheck Échoué",
                border_style="red",
            )
        )
        return "HOLD"

    # Priority handling: T212 execution overrides internal simulation
    if is_t212:
        is_simulation = False
        mode_text = "TRADING 212 EXECUTION"
    else:
        mode_text = "SIMULATION (1000€)" if is_simulation else "ANALYSIS"

    console.print(
        Panel(
            f"[bold blue]Trading AI {mode_text} for {ticker}[/bold blue]",
            border_style="blue",
        )
    )

    try:
        # Initialize the system
        system = EnhancedTradingSystem(
            ticker=ticker,
            initial_portfolio_value=1000 if (is_simulation or is_t212) else 10000,
        )

        # Run full analysis
        results, report = system.run_enhanced_analysis(is_simulation=is_simulation)

        # CLEAR OUTPUT OF DECISION
        decision = results["enhanced_decision"]
        risk = results["risk_metrics"]

        # Use the risk-adjusted signal for execution
        signal = results.get("risk_adjusted_signal", decision.final_signal)
        confidence = decision.final_confidence
        risk_level = risk.risk_level.name

        # T212 Execution
        if is_t212:
            from t212_executor import get_t212_ticker

            t212_key = get_t212_ticker(ticker)
            t212_state = load_t212_state(t212_key)

            # --- AJOUT : Récupération de l'état de position pour le Risk Manager ---
            is_holding = t212_state.get("active_position") is not None
            entry_price_index = t212_state.get("active_position", {}).get("entry_price_index") if is_holding else None

            # Recalculer le signal avec la conscience de la position
            # On ré-interroge le risk manager avec les infos de position
            signal, adjustment_reason = system.risk_manager.get_risk_adjusted_signal(
                decision.final_signal,
                decision.final_confidence,
                risk,
                price_data=results["market_data"].get(
                    "price_series"
                ),  # Passé par system.perform_enhanced_analysis si on le modifie
                ticker=ticker,
                is_holding=is_holding,
                entry_price_index=entry_price_index,
            )

            if signal != decision.final_signal:
                console.print(
                    f"[bold orange3]⚠️ Risk Management Override: {decision.final_signal} -> {signal}[/bold orange3]"
                )
                if "INERTIA" in adjustment_reason:
                    console.print(f"[bold cyan]ℹ️ {adjustment_reason}[/bold cyan]")

            if signal in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
                exec_signal = "BUY" if "BUY" in signal else "SELL"

                # --- Safety: cancel check + per-ticker lock ---
                # If the cycle has been cancelled (timeout from main loop),
                # do NOT place a real T212 order — the user has already been
                # told "HOLD appliqué". This prevents orphan threads from
                # executing real-money trades after the panel has been shown.
                if cancel_event is not None and cancel_event.is_set():
                    logger.warning(
                        f"⏱ Cycle for {ticker} was cancelled — skipping T212 {exec_signal} "
                        f"(original signal was {signal}) to avoid orphan-thread trade"
                    )
                    console.print(
                        f"[bold orange3]⏱ T212 {exec_signal} SKIPPED: cycle was cancelled by timeout[/bold orange3]"
                    )
                else:
                    ticker_lock = _get_ticker_lock(ticker)
                    with ticker_lock:
                        # Re-check cancel_event after acquiring the lock —
                        # protects against the case where the lock was held
                        # by an orphan from a previous cycle that just released it.
                        if cancel_event is not None and cancel_event.is_set():
                            logger.warning(f"⏱ Cancel detected after lock — skipping T212 {exec_signal}")
                        else:
                            console.print(
                                f"[bold yellow]🚀 Execution of the signal on Trading 212 for {ticker}... (original: {signal})[/bold yellow]"
                            )
                            execute_t212_trade(
                                exec_signal,
                                confidence,
                                ticker=ticker,
                                analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                signal_source="IA_HYBRID_T212",
                            )
            else:
                console.print(f"[bold blue]ℹ️ No trade executed (Signal is {signal})[/bold blue]")

        # --- AJOUT : Journalisation CSV pour débriefing détaillé ---
        journal_file = "trading_journal.csv"
        file_exists = Path(journal_file).exists()

        with open(journal_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Création de l'en-tête dynamique basé sur les modèles présents
            header = [
                "Timestamp",
                "Ticker",
                "FINAL_SIGNAL",
                "Confidence",
                "Risk_Level",
                "Risk_Adjusted",
                "T212_Capital",
            ]
            # Ajouter des colonnes pour chaque modèle possible
            model_names = [
                "classic",
                "llm_text",
                "llm_visual",
                "sentiment",
                "timesfm",
                "tensortrade",
                "vincent_ganne",
            ]
            for m in model_names:
                header.append(f"Model_{m}")

            if not file_exists:
                writer.writerow(header)

            t212_key = get_t212_ticker(ticker) if is_t212 else ticker
            t212_state = load_t212_state(t212_key, sync=False)
            capital_val = t212_state.get("current_capital", 1000.0)

            # Préparation de la ligne de données
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker,
                decision.final_signal,
                f"{confidence:.2%}",
                risk_level,
                signal,  # Le signal réellement exécuté (après risk management)
                f"{capital_val:.2f} €",
            ]

            # Dictionnaire des décisions pour un mapping facile
            dec_map = {d.model_name: f"{d.signal}({d.confidence:.2f})" for d in decision.individual_decisions}
            for m in model_names:
                row.append(dec_map.get(m, "N/A"))

            writer.writerow(row)
        # ------------------------------------------------

        # Color coding
        color = "green" if "BUY" in signal else "red" if "SELL" in signal else "yellow"

        # Final Summary Panel
        summary_table = Table(box=None)
        summary_table.add_column("Property", style="dim")
        summary_table.add_column("Value")

        summary_table.add_row("TICKER", f"[bold]{ticker}[/bold]")
        summary_table.add_row("FINAL DECISION", f"[bold {color}]{signal}[/bold {color}]")
        summary_table.add_row("CONFIDENCE", f"{confidence:.2%}")
        summary_table.add_row("RISK LEVEL", f"{risk_level}")

        if is_simulation:
            # Show current simulation state
            state = get_latest_portfolio_state(ticker)
            last_tx = get_latest_transaction(ticker)

            if state:
                summary_table.add_row("---", "---")
                summary_table.add_row("PORTFOLIO VALUE", f"[bold]{state[2]:.2f} €[/bold]")
                summary_table.add_row("CASH", f"{state[1]:.2f} €")
                summary_table.add_row("SHARES", f"{state[0]:.4f}")
                if last_tx:
                    summary_table.add_row("LAST TRADE", f"{last_tx[1]} on {last_tx[0]}")
        elif is_t212:
            from t212_executor import get_t212_ticker

            t212_key = get_t212_ticker(ticker)
            t212_state = load_t212_state(t212_key, sync=False)
            cap_val = t212_state.get("current_capital", 1000.0)
            pl_val = t212_state.get("total_realized_pl", 0.0)
            active_pos = t212_state.get("active_position")

            summary_table.add_row("---", "---")
            summary_table.add_row("T212 CAPITAL", f"[bold]{cap_val:.2f} €[/bold]")
            summary_table.add_row("T212 P/L", f"{pl_val:+.2f} €")
            if active_pos:
                summary_table.add_row("T212 POSITION", f"{active_pos['quantity']} shares")
        else:
            summary_table.add_row("REC. POSITION", f"${results['position_sizing'].recommended_size:,.2f}")

        console.print(
            Panel(
                summary_table,
                title=f"🎯 [bold]TRADING SIGNAL: {ticker}[/bold]",
                border_style=color,
                expand=False,
            )
        )

        return signal

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        console.print(f"[bold red]Error during analysis for {ticker}: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        return "ERROR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trading AI Analysis")
    parser.add_argument(
        "--ticker",
        type=str,
        nargs="+",
        default=["CRUDP.PA", "SXRV.DE"],
        help="Ticker(s) to analyze (default: CRUDP.PA and SXRV.DE)",
    )
    parser.add_argument(
        "--simul",
        action="store_true",
        help="Run in simulation mode (1000€ starting capital)",
    )
    parser.add_argument(
        "--t212",
        action="store_true",
        help="Execute trades on Trading 212 account (starts with 1000€ budget per ticker)",
    )
    args = parser.parse_args()

    import time

    start_time = time.time()

    for t in args.ticker:
        ticker_start = time.time()
        # Per-ticker cancel event: set on cycle timeout so the orphan worker
        # can bail out before placing a real T212 order. Persisted across
        # cycles so an orphan from cycle N is still visible to cycle N+1.
        cancel_event = _get_cancel_event(t)
        cancel_event.clear()  # Reset for this cycle

        # Per-ticker lock prevents two concurrent run_trading_analysis on the
        # same ticker (orphan thread from cycle N + new thread from cycle N+1).
        # IMPORTANT: We do NOT use `with ThreadPoolExecutor(...)` because
        # __exit__ calls shutdown(wait=True) which would block on timeout.
        # Instead we explicitly shutdown(wait=False) so the next ticker can
        # proceed immediately. The orphan thread keeps running but is gated
        # by the cancel_event check + the per-ticker lock before any T212 trade.
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"cycle_{t}")
        future = ex.submit(run_trading_analysis, t, args.simul, args.t212, cancel_event)
        try:
            future.result(timeout=CYCLE_TIMEOUT_SECONDS)
            ex.shutdown(wait=True)
        except FuturesTimeoutError:
            elapsed = time.time() - ticker_start
            cancel_event.set()  # Signal the orphan worker to bail before T212 trade
            logger.error(
                f"Cycle timeout ({CYCLE_TIMEOUT_SECONDS}s) reached for {t} after {elapsed:.1f}s — "
                f"cancel_event set; orphan thread will skip T212 execution"
            )
            ex.shutdown(wait=False)  # Do NOT block — daemon thread keeps running
            console = Console()
            console.print(
                Panel(
                    f"[bold red]⏱ Cycle timeout for {t} ({elapsed:.1f}s > {CYCLE_TIMEOUT_SECONDS}s)[/bold red]\n"
                    f"Le cycle a été interrompu. Le signal HOLD est appliqué par défaut.\n"
                    f"L'event d'annulation a été armé : le thread orphelin ne pourra\n"
                    f"pas passer d'ordre T212 réel même s'il termine l'analyse.",
                    title="Cycle Timeout",
                    border_style="red",
                )
            )
        except Exception as cycle_exc:
            logger.error(f"Unexpected error during cycle for {t}: {cycle_exc}")
            cancel_event.set()
            ex.shutdown(wait=False)

    duration = time.time() - start_time
    logging.info(f"Total execution time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
