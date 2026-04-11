from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from backtest_engine import Backtester

if __name__ == "__main__":
    # Test sur 2 mois récents
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    # Mode Fast pour ne pas attendre Ollama
    bt = Backtester(
        ticker_etf="SXRV.DE", 
        ticker_index="^NDX", 
        fast_mode=True,
        use_visual=False
    )
    bt.run(start, end)
