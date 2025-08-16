import pandas as pd
import numpy as np

def _calculate_performance_metrics(df: pd.DataFrame) -> dict:
    """Calcul des métriques de performance avancées"""
    strategy_returns = df['Strategy_Returns'].dropna()
    benchmark_returns = df['Returns']

    # Rendements annualisés
    strategy_annual = strategy_returns.mean() * 252
    benchmark_annual = benchmark_returns.mean() * 252

    # Volatilités annualisées
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)

    # Ratios de Sharpe
    risk_free_rate = 0.02  # 2% taux sans risque
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

def run_backtest(data: pd.DataFrame, signals: pd.Series, transaction_cost_pct: float = 0.001) -> tuple[pd.DataFrame, dict]:
    """
    Exécute un backtest basé sur une série de signaux, en incluant les coûts de transaction.
    signals: une série avec 1 pour long, 0 pour neutre.
    transaction_cost_pct: le coût d'un aller-retour en pourcentage (ex: 0.001 pour 0.1%).
    """
    df = data.copy()

    # La série 'signals' représente la position désirée (1 ou 0)
    df['Position'] = signals.shift(1).fillna(0) # On entre à la bougie suivante

    # Identifier les jours de trade (changement de position)
    df['Trades'] = df['Position'].diff().abs()

    # Calculer les coûts de transaction
    df['Transaction_Costs'] = df['Trades'] * transaction_cost_pct

    # Calculer les rendements de la stratégie en déduisant les coûts
    df['Strategy_Returns'] = (df['Returns'] * df['Position']) - df['Transaction_Costs']

    # Calculer les rendements cumulés
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

    # Métriques de performance
    performance_metrics = _calculate_performance_metrics(df)

    # Ajouter le coût total au rapport
    performance_metrics['total_transaction_costs'] = df['Transaction_Costs'].sum()

    return df, performance_metrics
