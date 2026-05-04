"""
TradingAI Framework Algorithm for QuantConnect Lean.

Full Alpha-Portfolio-Risk-Execution framework using Trading-AI's
composite Alpha Models. This algorithm demonstrates the institutional-grade
backtesting approach with realistic T212 fee models and slippage.
"""

from AlgorithmImports import *
from AlphaModels.TradingAICompositeAlpha import TradingAICompositeAlpha


class TradingAIFrameworkAlgorithm(QCAlgorithm):

    def initialize(self):
        start_year = int(self.get_parameter("start-year", 2020))
        end_year = int(self.get_parameter("end-year", 2025))
        cash = float(self.get_parameter("cash", 1000))
        max_drawdown = float(self.get_parameter("max-drawdown", 0.15))

        self.set_start_date(start_year, 1, 1)
        self.set_end_date(end_year, 12, 31)
        self.set_cash(cash)
        self.set_account_currency("EUR")

        symbols = [
            Symbol.create("QQQ", SecurityType.EQUITY, Market.USA),
            Symbol.create("USO", SecurityType.EQUITY, Market.USA),
        ]

        self.set_universe_selection(ManualUniverseSelectionModel(symbols))

        self.set_alpha(TradingAICompositeAlpha())

        self.set_portfolio_construction(
            EqualWeightingPortfolioConstructionModel(Resolution.DAILY)
        )

        self.set_risk_management(
            MaximumDrawdownPercentPerSecurity(max_drawdown)
        )

        self.set_execution(ImmediateExecutionModel())

        for symbol in symbols:
            security = self.securities[symbol]
            security.set_fee_model(ConstantFeeModel(0.001))
            security.set_slippage_model(VolumeShareSlippageModel(0.025, 0.05))

        self.set_warm_up(60, Resolution.DAILY)

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED:
            self.log(
                f"ORDER: {order_event.symbol} {order_event.direction} "
                f"@ {order_event.fill_price:.2f} x{order_event.fill_quantity}"
            )

    def on_end_of_algorithm(self):
        total_value = self.portfolio.total_portfolio_value
        initial_cash = float(self.get_parameter("cash", 1000))
        total_return = (total_value / initial_cash) - 1

        self.log(f"=== FRAMEWORK RESULTS ===")
        self.log(f"Portfolio Value: {total_value:.2f} EUR")
        self.log(f"Total Return: {total_return:.2%}")
        self.log(f"Total Trades: {self.transactions.orders_count}")
