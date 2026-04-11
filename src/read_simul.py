"""
Helper script to read the simulation history from the database.
Usage: uv run python src/read_simul.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from database import get_portfolio_history, get_transactions_history
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def show_simulation_report():
    console = Console()
    
    # Transactions
    tx_df = get_transactions_history()
    
    # Portfolio
    port_df = get_portfolio_history()
    
    console.print(Panel("[bold cyan]SIMULATION TRADING REPORT[/bold cyan]", expand=False))
    
    # Current State
    if not port_df.empty:
        latest = port_df.iloc[-1]
        summary = Table(title="Current Simulation State")
        summary.add_column("Ticker")
        summary.add_column("Total Value")
        summary.add_column("Cash")
        summary.add_column("Shares Held")
        summary.add_column("Return")
        
        # Calculate return from initial 1000
        initial_val = 1000.0 # Standard for our simulation
        total_return = (latest['total_value'] - initial_val) / initial_val
        
        summary.add_row(
            latest['ticker'],
            f"{latest['total_value']:.2f} €",
            f"{latest['cash']:.2f} €",
            f"{latest['position']:.4f}",
            f"{total_return:+.2%}"
        )
        console.print(summary)
    
    # Transaction History
    if not tx_df.empty:
        table = Table(title="Transaction History (Last 10)")
        table.add_column("Date")
        table.add_column("Type")
        table.add_column("Ticker")
        table.add_column("Price")
        table.add_column("Qty")
        table.add_column("Total Cost")
        
        # Last 10 trades
        last_tx = tx_df.tail(10)
        for _, row in last_tx.iterrows():
            color = "green" if row['type'] == 'BUY' else "red"
            table.add_row(
                str(row['date']),
                f"[{color}]{row['type']}[/{color}]",
                row['ticker'],
                f"{row['price']:.2f} €",
                f"{row['quantity']:.4f}",
                f"{row['cost']:.2f} €"
            )
        console.print(table)
    else:
        console.print("[yellow]No transactions yet.[/yellow]")

if __name__ == "__main__":
    show_simulation_report()
