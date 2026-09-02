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
from typing import Any
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
from llm_client import check_ai_health
from bootstrap import setup_environment

# Load environment
load_dotenv()

# Setup logging
setup_environment("trading.log")
logger = logging.getLogger("TradingAI")

# Cycle timeout: 40 minutes per ticker. Prevents infinite hangs on LLM calls.
CYCLE_TIMEOUT_SECONDS = 40 * 60

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
    """Vérifie si TimesFM 3.0 (package timesfm3) est installé.

    Utilise find_spec (sans importer torch) : le chargement réel du modèle
    reste dans src/timesfm_model.py avec ses fallbacks.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("timesfm3")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        console = Console()
        console.print(
            Panel(
                "[bold red]ERREUR : TimesFM 3.0 (package timesfm3) n'est pas installé.[/bold red]\n\n"
                "[*] Veuillez lancer la commande suivante pour tout configurer automatiquement :\n"
                "    [bold cyan]uv sync[/bold cyan]",
                title="Setup Manquant",
                border_style="red",
            )
        )
        return False
    return True





def _execute_t212_orders(
    ticker: str,
    system: EnhancedTradingSystem,
    decision: Any,
    risk: Any,
    results: dict,
    cancel_event: threading.Event | None,
    console: Console,
) -> str:
    """Handle position checking, risk overrides and order execution on Trading 212."""
    from t212_executor import get_t212_ticker, INITIAL_BUDGETS, DEFAULT_INITIAL_BUDGET

    t212_key = get_t212_ticker(ticker)
    t212_state = load_t212_state(t212_key)

    is_holding = t212_state.get("active_position") is not None
    entry_price_index = (
        t212_state.get("active_position", {}).get("entry_price_index")
        if is_holding
        else None
    )

    signal, adjustment_reason = system.risk_manager.get_risk_adjusted_signal(
        decision.final_signal,
        decision.final_confidence,
        risk,
        price_data=results["market_data"].get("price_series"),
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

    if signal not in ["BUY", "STRONG_BUY", "SELL", "STRONG_SELL"]:
        console.print(f"[bold blue]ℹ️ No trade executed (Signal is {signal})[/bold blue]")
        return signal

    exec_signal = "BUY" if "BUY" in signal else "SELL"
    if cancel_event is not None and cancel_event.is_set():
        logger.warning(
            f"⏱ Cycle for {ticker} was cancelled — skipping T212 {exec_signal} "
            f"(original signal was {signal}) to avoid orphan-thread trade"
        )
        console.print(
            f"[bold orange3]⏱ T212 {exec_signal} SKIPPED: cycle was cancelled by timeout[/bold orange3]"
        )
        return signal

    ticker_lock = _get_ticker_lock(ticker)
    with ticker_lock:
        if cancel_event is not None and cancel_event.is_set():
            logger.warning(f"⏱ Cancel detected after lock — skipping T212 {exec_signal}")
            return signal

        console.print(
            f"[bold yellow]🚀 Execution of the signal on Trading 212 for {ticker}... (original: {signal})[/bold yellow]"
        )
        budget_ticker = INITIAL_BUDGETS.get(t212_key, DEFAULT_INITIAL_BUDGET)
        rec_eur = None
        try:
            rec_eur = results["position_sizing"].recommended_size
            sizing_ratio = max(0.3, min(rec_eur / budget_ticker, 1.0)) if budget_ticker > 0 else 0.75
        except (KeyError, AttributeError, TypeError):
            sizing_ratio = 0.75

        rec_str = f"{rec_eur:.0f}€" if rec_eur is not None else "?"
        logger.info(
            f"📏 Sizing: risk-manager recommended {rec_str} "
            f"-> sizing_ratio={sizing_ratio:.2f} (budget {budget_ticker}€)"
        )

        execute_t212_trade(
            exec_signal,
            decision.final_confidence,
            ticker=ticker,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            signal_source="IA_HYBRID_T212",
            sizing_ratio=sizing_ratio,
        )

    return signal


def _write_trading_journal(
    ticker: str,
    decision: Any,
    confidence: float,
    risk_level: str,
    signal: str,
    is_t212: bool,
) -> None:
    """Write trading analysis row to trading_journal.csv."""
    journal_file = "trading_journal.csv"
    file_exists = Path(journal_file).exists()

    with open(journal_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "Timestamp",
            "Ticker",
            "FINAL_SIGNAL",
            "Confidence",
            "Risk_Level",
            "Risk_Adjusted",
            "T212_Equity",
        ]
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

        from t212_executor import get_t212_ticker
        t212_key = get_t212_ticker(ticker) if is_t212 else ticker
        t212_state = load_t212_state(t212_key, sync=False)
        # GO-gate 7 (audit 2026-08-19): the old T212_Capital column mixed the
        # position value (when open) with cash (when flat), producing a fake
        # -71.6% drawdown. The equity (budget + realized + unrealized) is the
        # real per-ticker performance curve.
        capital_val = t212_state.get("equity", t212_state.get("current_capital", 1000.0))

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            decision.final_signal,
            f"{confidence:.2%}",
            risk_level,
            signal,
            f"{capital_val:.2f} €",
        ]

        dec_map = {d.model_name: f"{d.signal}({d.confidence:.2f})" for d in decision.individual_decisions}
        for m in model_names:
            row.append(dec_map.get(m, "N/A"))

        writer.writerow(row)


def _render_summary_panel(
    ticker: str,
    signal: str,
    confidence: float,
    risk_level: str,
    is_simulation: bool,
    is_t212: bool,
    results: dict,
    console: Console,
) -> None:
    """Render Rich summary table inside a panel."""
    color = "green" if "BUY" in signal else "red" if "SELL" in signal else "yellow"
    summary_table = Table(box=None)
    summary_table.add_column("Property", style="dim")
    summary_table.add_column("Value")

    summary_table.add_row("TICKER", f"[bold]{ticker}[/bold]")
    summary_table.add_row("FINAL DECISION", f"[bold {color}]{signal}[/bold {color}]")
    summary_table.add_row("CONFIDENCE", f"{confidence:.2%}")
    summary_table.add_row("RISK LEVEL", f"{risk_level}")

    if is_simulation:
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


def run_trading_analysis(
    ticker: str,
    is_simulation: bool = False,
    is_t212: bool = False,
    cancel_event: threading.Event | None = None,
):
    if not check_setup():
        return

    console = Console()

    # Vérification santé IA avant chaque cycle
    ai_ok = check_ai_health()
    if not ai_ok:
        logger.critical(f"AUCUN FOURNISSEUR IA CONFIGURÉ pour {ticker} — le cycle de trading est ignoré.")
        console.print(
            Panel(
                f"[bold red]AUCUN FOURNISSEUR IA CONFIGURÉ (NexusAI)[/bold red]\n\n"
                f"Le cycle pour {ticker} est ignoré. Les modèles LLM sont indispensables.\n"
                f"Vérifiez vos clés API dans le fichier .env.",
                title="AI Healthcheck Échoué",
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
        # Initialize the system.
        # In T212 execution mode, write_db=False: the analysis step must NOT
        # write phantom simulated trades to trading_history.db — only the
        # t212_executor writes, after a real broker-confirmed fill. This keeps
        # the DB as the single source of truth (broker), preventing the
        # persistent desync where DB showed trades the broker never executed.
        system = EnhancedTradingSystem(
            ticker=ticker,
            initial_portfolio_value=1000 if (is_simulation or is_t212) else 10000,
            write_db=not is_t212,
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
            signal = _execute_t212_orders(
                ticker=ticker,
                system=system,
                decision=decision,
                risk=risk,
                results=results,
                cancel_event=cancel_event,
                console=console,
            )

        # Journalisation CSV pour débriefing détaillé
        _write_trading_journal(
            ticker=ticker,
            decision=decision,
            confidence=confidence,
            risk_level=risk_level,
            signal=signal,
            is_t212=is_t212,
        )

        # Affichage du panneau de résumé
        _render_summary_panel(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            risk_level=risk_level,
            is_simulation=is_simulation,
            is_t212=is_t212,
            results=results,
            console=console,
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
