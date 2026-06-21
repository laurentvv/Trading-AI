import asyncio
from datetime import datetime, timedelta
import pandas as pd
import pytz
from ziplime.core.run_simulation import run_simulation
from ziplime.data.data_sources.yahoo_finance_data_source import YahooFinanceDataSource
from ziplime.assets.services.asset_service import AssetService
from ziplime.assets.repositories.sqlalchemy_asset_repository import SqlAlchemyAssetRepository
from ziplime.assets.repositories.sqlalchemy_adjustments_repository import SqlAlchemyAdjustmentRepository


def get_asset_service():
    db_url = "sqlite+aiosqlite:///ziplime_assets.db"
    asset_repo = SqlAlchemyAssetRepository(db_url, future_chain_predicates=[])
    adj_repo = SqlAlchemyAdjustmentRepository(db_url)
    return AssetService(asset_repo, adj_repo)


async def main():
    print("Starting ZipLime simulation")

    # Nous utilisons YahooFinanceDataSource pour telecharger les donnees automatiquement.
    # Ceci est compatible avec ZipLime
    source = YahooFinanceDataSource()
    asset_service = get_asset_service()

    # Lire logs_prod/trading_journal.csv si disponible
    import os

    if os.path.exists("logs_prod/trading_journal.csv"):
        journal = pd.read_csv("logs_prod/trading_journal.csv")
        journal["date"] = pd.to_datetime(journal["date"])
        min_dt = journal["date"].min()
        max_dt = journal["date"].max()
        start_date = pytz.UTC.localize(min_dt)
        end_date = pytz.UTC.localize(max_dt)
    else:
        start_date = pytz.UTC.localize(datetime(2023, 1, 1))
        end_date = pytz.UTC.localize(datetime(2023, 1, 31))

    import ziplime.utils.run_algo as ra

    # Patch the run_algo internal benchmark logic so we can bypass the SPY missing bug
    old_prepare = ra._prepare_algorithm

    async def mock_prepare(*args, **kwargs):
        kwargs["benchmark_asset_symbol"] = None
        import ziplime.sources.benchmark_source as bs

        async def skip_val(*a, **kw):
            pass

        bs.BenchmarkSource.validate_benchmark = skip_val
        return await old_prepare(*args, **kwargs)

    ra._prepare_algorithm = mock_prepare

    try:
        coro = run_simulation(
            start_date=start_date,
            end_date=end_date,
            trading_calendar="XNYS",
            emission_rate=timedelta(days=1),
            total_cash=1000.0,
            market_data_source=source,
            custom_data_sources=[],
            algorithm_file="backtest_ziplime_algo.py",
            stop_on_error=False,
            asset_service=asset_service,
            benchmark_asset_symbol=None,
        )
        results = await coro

        if results.errors:
            print("Errors occurred during backtest.")
            for err in results.errors[:2]:
                print(f"Error {err.dt}: {err.error}")

        if hasattr(results, "perf"):
            perf_df = results.perf
            print(f"Performance shape: {perf_df.shape}")
            print(f"Final Portfolio Value: {perf_df.iloc[-1]['portfolio_value']:.2f} EUR")

            # Save results
            if not os.path.exists("logs_prod"):
                os.makedirs("logs_prod")
            perf_df.to_csv("logs_prod/ziplime_perf.csv")
            print("Saved logs_prod/ziplime_perf.csv")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print("Run Simulation failed:", e)


if __name__ == "__main__":
    asyncio.run(main())
