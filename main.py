"""
Trading AI - Unified Entry Point
This script performs a full analysis: data fetching, model training, and hybrid decision making.
Usage: python main.py --ticker QQQ
"""

import logging
import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import system modules
from enhanced_trading_example import EnhancedTradingSystem
from t212_executor import execute_t212_trade, load_portfolio_state as load_t212_state
from database import get_latest_portfolio_state, get_latest_transaction

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingAI")

def check_setup() -> bool:
    """Vérifie si TimesFM 2.5 est correctement installé et patché"""
    vendor_path = Path(__file__).parent / "vendor" / "timesfm"
    if not vendor_path.exists():
        console = Console()
        console.print(Panel(
            "[bold red]ERREUR : TimesFM 2.5 n'est pas installé.[/bold red]\n\n"
            "[*] Veuillez lancer la commande suivante pour tout configurer automatiquement :\n"
            "    [bold cyan]uv run setup[/bold cyan]",
            title="Setup Manquant",
            border_style="red"
        ))
        return False
    return True

def run_trading_analysis(ticker: str, is_simulation: bool = False, is_t212: bool = False):
    if not check_setup():
        return
        
    console = Console()
    
    # Priority handling: T212 execution overrides internal simulation
    if is_t212:
        is_simulation = False
        mode_text = "TRADING 212 EXECUTION"
    else:
        mode_text = "SIMULATION (1000€)" if is_simulation else "ANALYSIS"
        
    console.print(Panel(f"[bold blue]Trading AI {mode_text} for {ticker}[/bold blue]", border_style="blue"))
    
    try:
        # Initialize the system
        system = EnhancedTradingSystem(ticker=ticker, initial_portfolio_value=1000 if (is_simulation or is_t212) else 10000)
        
        # Run full analysis
        results, report = system.run_enhanced_analysis(is_simulation=is_simulation)
        
        # CLEAR OUTPUT OF DECISION
        decision = results['enhanced_decision']
        risk = results['risk_metrics']
        
        # Use the risk-adjusted signal for execution
        signal = results.get('risk_adjusted_signal', decision.final_signal)
        confidence = decision.final_confidence
        risk_level = risk.risk_level.name
        
        # T212 Execution
        if is_t212:
            if signal != decision.final_signal:
                console.print(f"[bold orange3]⚠️ Risk Management Override: {decision.final_signal} -> {signal}[/bold orange3]")
            
            if signal in ["BUY", "SELL"]:
                console.print(f"[bold yellow]🚀 Execution of the signal on Trading 212 for {ticker}...[/bold yellow]")
                execute_t212_trade(signal, confidence, ticker=ticker)
            else:
                console.print(f"[bold blue]ℹ️ No trade executed (Signal is {signal})[/bold blue]")

        # --- AJOUT : Journalisation CSV pour débriefing ---
        journal_file = "trading_journal.csv"
        file_exists = Path(journal_file).exists()

        with open(journal_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Ticker", "Signal", "Confidence", "Risk", "LLM_Analysis", "Capital_T212"])

            t212_ticker = ticker.split('.')[0]
            t212_state = load_t212_state(t212_ticker)
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker,
                signal,
                f"{confidence:.2%}",
                risk_level,
                decision.individual_decisions[1].analysis if len(decision.individual_decisions) > 1 else "N/A",
                f"{t212_state['current_capital']:.2f} €"
            ])
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
            t212_ticker = ticker.split('.')[0]
            t212_state = load_t212_state(t212_ticker)
            summary_table.add_row("---", "---")
            summary_table.add_row("T212 CAPITAL", f"[bold]{t212_state['current_capital']:.2f} €[/bold]")
            summary_table.add_row("T212 P/L", f"{t212_state['total_realized_pl']:.2f} €")
            if t212_state['active_position']:
                pos = t212_state['active_position']
                summary_table.add_row("T212 POSITION", f"{pos['quantity']} shares")
        else:
            summary_table.add_row("REC. POSITION", f"${results['position_sizing'].recommended_size:,.2f}")
        
        console.print(Panel(
            summary_table,
            title=f"🎯 [bold]TRADING SIGNAL: {ticker}[/bold]",
            border_style=color,
            expand=False
        ))
        
        return signal
        
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        console.print(f"[bold red]Error during analysis for {ticker}: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return "ERROR"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Trading AI Analysis')
    parser.add_argument('--ticker', type=str, nargs='+', default=['CRUDP.PA', 'SXRV.DE'], help='Ticker(s) to analyze (default: CRUDP.PA and SXRV.DE)')
    parser.add_argument('--simul', action='store_true', help='Run in simulation mode (1000€ starting capital)')
    parser.add_argument('--t212', action='store_true', help='Execute trades on Trading 212 account (starts with 1000€ budget per ticker)')
    args = parser.parse_args()
    
    for t in args.ticker:
        run_trading_analysis(t, args.simul, args.t212)
