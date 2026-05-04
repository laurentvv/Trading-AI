"""
TradingAI Baseline Algorithm for QuantConnect Lean.

Buy-and-hold strategy using US proxies for Trading-AI's European ETFs.
Includes T212-realistic fee model (0.1% per trade) and volume-share slippage.
Serves as benchmark for comparing against Trading-AI's hybrid signals.

Usage: lean backtest
"""

from AlgorithmImports import *


class TradingAIBaseline(QCAlgorithm):

    def initialize(self):
        start_year = self.get_parameter("start-year", 2020)
        end_year = self.get_parameter("end-year", 2025)
        cash = float(self.get_parameter("cash", 1000))

        self.set_start_date(start_year, 1, 1)
        self.set_end_date(end_year, 12, 31)
        self.set_cash(cash)
        self.set_account_currency("EUR")

        self.nasdaq = self.add_equity("QQQ", Resolution.DAILY)
        self.oil = self.add_equity("USO", Resolution.DAILY)
        self.set_benchmark("QQQ")

        self.rsi = self.rsi("QQQ", 14)
        self.macd = self.macd("QQQ", 12, 26, 9)
        self.bollinger = self.bb("QQQ", 20, 2)
        self.atr = self.atr("QQQ", 14)

        for security in [self.nasdaq, self.oil]:
            security.set_fee_model(ConstantFeeModel(0.001))
            security.set_slippage_model(VolumeShareSlippageModel(0.025, 0.05))

        self.set_warm_up(50, Resolution.DAILY)

        self._rebalanced = False

    def on_data(self, data):
        if self.is_warming_up:
            return

        if not self._rebalanced:
            self.set_holdings("QQQ", 0.50)
            self.set_holdings("USO", 0.50)
            self._rebalanced = True

    def on_end_of_algorithm(self):
        self.log(f"=== BASELINE RESULTS ===")
        self.log(f"Portfolio Value: {self.portfolio.total_portfolio_value:.2f} EUR")
        self.log(f"Total Return: {self.portfolio.total_portfolio_value / 1000 - 1:.2%}")
        self.log(f"QQQ Holdings: {self.portfolio['QQQ'].quantity}")
        self.log(f"USO Holdings: {self.portfolio['USO'].quantity}")
