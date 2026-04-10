"""
Trading AI - Unified Entry Point
This script performs a full analysis: data fetching, model training, and hybrid decision making.
Usage: python main.py --ticker QQQ
"""

import logging
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import system modules
from enhanced_trading_example import EnhancedTradingSystem

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

def run_trading_analysis(ticker: str, is_simulation: bool = False):
    console = Console()
    mode_text = "SIMULATION (1000€)" if is_simulation else "ANALYSIS"
    console.print(Panel(f"[bold blue]Trading AI {mode_text} for {ticker}[/bold blue]", border_style="blue"))
    
    try:
        # Initialize the system
        # If simulation, we use fixed capital of 1000 in the logic
        system = EnhancedTradingSystem(ticker=ticker, initial_portfolio_value=1000 if is_simulation else 10000)
        
        # Run full analysis with simulation flag
        results, report = system.run_enhanced_analysis(is_simulation=is_simulation)
        
        # CLEAR OUTPUT OF DECISION
        decision = results['enhanced_decision']
        risk = results['risk_metrics']
        
        signal = decision.final_signal
        confidence = decision.final_confidence
        risk_level = risk.risk_level.name
        
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
            from database import get_latest_portfolio_state, get_latest_transaction
            state = get_latest_portfolio_state(ticker)
            last_tx = get_latest_transaction(ticker)
            
            summary_table.add_row("---", "---")
            summary_table.add_row("PORTFOLIO VALUE", f"[bold]{state[2]:.2f} €[/bold]")
            summary_table.add_row("CASH", f"{state[1]:.2f} €")
            summary_table.add_row("SHARES", f"{state[0]:.4f}")
            if last_tx:
                summary_table.add_row("LAST TRADE", f"{last_tx[1]} on {last_tx[0]}")

        else:
            summary_table.add_row("REC. POSITION", f"${results['position_sizing'].recommended_size:,.2f}")
        
        console.print(Panel(
            summary_table,
            title="🎯 [bold]TRADING SIGNAL[/bold]",
            border_style=color,
            expand=False
        ))
        
        return signal
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        console.print(f"[bold red]Error during analysis: {e}[/bold red]")
        return "ERROR"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Trading AI Analysis')
    parser.add_argument('--ticker', type=str, default='QQQ', help='Ticker to analyze (default: QQQ)')
    parser.add_argument('--simul', action='store_true', help='Run in simulation mode (1000€ starting capital)')
    parser.add_argument('--t212', action='store_true', help='Execute trades on Trading 212 account (starts with 1000€ budget)')
    args = parser.parse_args()
    
    run_trading_analysis(args.ticker, args.simul, args.t212)
