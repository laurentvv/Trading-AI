# Active Context

## Current Status
The project has undergone a major simplification and optimization phase. The complex automated scheduler has been removed in favor of a powerful, unified CLI tool (`main.py`) that can be run on-demand or via simple OS-level scheduling (cron/Task Scheduler).

## Recent Accomplishments
- **Gemma 4 Migration**: Upgraded cognitive analysis to `gemma4:e4b` for superior reasoning and technical/fundamental synthesis.
- **Hyperliquid Integration**: Added decentralized sentiment analysis (Funding Rate, Open Interest) for the Oil/WTI strategy, providing a unique on-chain contrarian signal.
- **Robustness & Windows Compatibility**:
    - Implemented **UTF-8 logging** to prevent `UnicodeEncodeError` with emojis on Windows terminals.
    - Fixed Pandas `FutureWarning` in the Vincent Ganne model and `KeyError` in `main.py` state loading.
- **T212 Ticker Certification**: Finalized and verified exact instrument identifiers for Trading 212 (`SXRVd_EQ` for Nasdaq EUR and `CRUDl_EQ` for WTI Oil).
- **Risk-Adjusted Execution**: The system now strictly follows the `AdvancedRiskManager` filtered signal (Accuracy First).

## Next Steps
- **Oil Strategy Validation**: Evaluate how Hyperliquid's *Funding Rate* influences Gemma 4's decisions during volatile Oil sessions.
- **Demo Mode Monitoring**: Continuous execution via `schedule.py` on the T212 Demo account.
- **Sentiment Refinement**: Further tune the weight between Alpha Vantage (ticker-specific) and AlphaEar (macro trends) sentiment scores.

## Active Decisions
- **Manual Control**: Shifted from an "automated agent" philosophy to a "decision support tool" philosophy to give the user more control and reduce complexity.
- **Local AI Priority**: Maintained the requirement for local Ollama/**Gemma 4** to ensure data privacy and zero cost.
