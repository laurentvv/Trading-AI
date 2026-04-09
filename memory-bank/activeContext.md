# Active Context

## Current Status
The project has undergone a major simplification and optimization phase. The complex automated scheduler has been removed in favor of a powerful, unified CLI tool (`main.py`) that can be run on-demand or via simple OS-level scheduling (cron/Task Scheduler).

## Recent Accomplishments
- **Dependency Migration**: Fully migrated to `uv`. Python 3.12 is now the standard to ensure compatibility with `TimesFM` and `JAX`.
- **Unified Engine**: Consolidated all logic into `src/enhanced_trading_example.py` and provided a single entry point via root `main.py`.
- **Simulation Mode**: Implemented a strict Paper Trading mode (`--simul`) with persistent SQLite tracking and 1000 € starting capital.
- **Model Fixes**:
    - Fixed `TimesFM` initialization and checkpoint path issues.
    - Updated `Gemma 3` prompt engineering for reliable JSON output.
    - Optimized `ClassicModel` with `TimeSeriesSplit` and `ffill` to eliminate data leakage.
- **Cleanup**: Removed over 20 redundant files, legacy scripts, and outdated databases.

## Next Steps
- **1-Month Simulation Test**: Run the system daily with `--simul` to evaluate real-time performance.
- **Reporting Improvements**: Enhance `src/read_simul.py` with more detailed P&L charts if needed.
- **Production Integration**: Explore Trading 212 API for transition from simulation to real execution.

## Active Decisions
- **Manual Control**: Shifted from an "automated agent" philosophy to a "decision support tool" philosophy to give the user more control and reduce complexity.
- **Local AI Priority**: Maintained the requirement for local Ollama/Gemma 3 to ensure data privacy and zero cost.
