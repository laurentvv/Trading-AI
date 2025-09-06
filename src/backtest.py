import pandas as pd
import numpy as np
# --- NEW: Import database module ---
from database import insert_transaction, insert_portfolio_state

def _calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calculates advanced performance metrics."""
    strategy_returns = df['Strategy_Returns'].dropna()
    benchmark_returns = df['Returns']

    # Annualized returns
    strategy_annual = strategy_returns.mean() * 252
    benchmark_annual = benchmark_returns.mean() * 252

    # Annualized volatilities
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)

    # Sharpe ratios
    risk_free_rate = 0.02  # 2% risk-free rate
    strategy_sharpe = (strategy_annual - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = (benchmark_annual - risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    win_rate = len(winning_trades) / len(strategy_returns[strategy_returns != 0]) if len(strategy_returns[strategy_returns != 0]) > 0 else 0

    # Calmar ratio
    calmar_ratio = strategy_annual / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'strategy_annual_return': strategy_annual,
        'benchmark_annual_return': benchmark_annual,
        'strategy_volatility': strategy_vol,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'calmar_ratio': calmar_ratio,
        'total_trades': len(strategy_returns[strategy_returns != 0]),
        'final_portfolio_value': df['Cumulative_Strategy'].iloc[-1]
    }

def run_backtest(data: pd.DataFrame, signals: pd.Series, transaction_cost_pct: float = 0.001, ticker: str = 'QQQ') -> tuple[pd.DataFrame, dict]:
    """
    Runs a backtest based on a series of signals, including transaction costs.
    signals: a series with 1 for long, 0 for neutral.
    transaction_cost_pct: the cost of a round trip as a percentage (e.g., 0.001 for 0.1%).
    ticker: The ticker symbol for database logging.
    """
    df = data.copy()

    # The 'signals' series represents the desired position (1 or 0)
    df['Position'] = signals.shift(1).fillna(0) # We enter on the next candle

    # Identify trade days (change in position)
    df['Trades'] = df['Position'].diff().abs()

    # Calculate transaction costs
    df['Transaction_Costs'] = df['Trades'] * transaction_cost_pct

    # Calculate strategy returns minus costs
    df['Strategy_Returns'] = (df['Returns'] * df['Position']) - df['Transaction_Costs']

    # Calculate cumulative returns
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    
    # --- NEW: Log transactions and portfolio state to database ---
    # Initialize portfolio values
    initial_cash = 10000.0 # Example starting cash
    cash = initial_cash
    position = 0
    quantity_held = 0.0
    
     # Get all trade dates
    trade_dates = df[df['Trades'] > 0].index
    
    for date in df.index:
        # Get price for the day (using Close price)
        price = df.loc[date, 'Close']
        
        # Check if it's a trade day
        if date in trade_dates:
            desired_position = df.loc[date, 'Position']
            current_position = 0 if date == df.index[0] else df.loc[df.index[df.index.get_loc(date)-1], 'Position']
            
            # BUY signal (transition from 0 to 1)
            if desired_position == 1 and current_position == 0:
                # Calculate quantity to buy (use all cash)
                quantity_to_buy = cash / price
                cost = quantity_to_buy * price * transaction_cost_pct
                cash -= (quantity_to_buy * price + cost)
                quantity_held = quantity_to_buy
                
                 # Log transaction
                insert_transaction(date.strftime('%Y-%m-%d'), ticker, 'BUY', quantity_to_buy, price, cost, 'backtest', 'Position opened')
                
            # SELL signal (transition from 1 to 0)
            elif desired_position == 0 and current_position == 1:
                # Sell all held quantity
                quantity_to_sell = quantity_held
                cost = quantity_to_sell * price * transaction_cost_pct
                cash += (quantity_to_sell * price - cost)
                quantity_held = 0.0
                
                # Log transaction
                insert_transaction(date.strftime('%Y-%m-%d'), ticker, 'SELL', quantity_to_sell, price, cost, 'backtest', 'Position closed')
        
        # Update portfolio state for the day
        total_value = cash + (quantity_held * price)
        benchmark_value = df.loc[date, 'Cumulative_Returns'] * initial_cash # Assuming initial investment of initial_cash in benchmark
        
        # Log portfolio state
        insert_portfolio_state(date.strftime('%Y-%m-%d'), ticker, quantity_held, cash, total_value, benchmark_value)
        
        # Update df with portfolio values for plotting etc. (optional, but useful)
        df.loc[date, 'Portfolio_Cash'] = cash
        df.loc[date, 'Portfolio_Position'] = quantity_held
        df.loc[date, 'Portfolio_Total_Value'] = total_value
        df.loc[date, 'Benchmark_Value'] = benchmark_value
    # --- END NEW ---

    # Performance metrics
    performance_metrics = _calculate_performance_metrics(df)

    # Add total cost to the report
    performance_metrics['total_transaction_costs'] = df['Transaction_Costs'].sum()

    return df, performance_metrics
