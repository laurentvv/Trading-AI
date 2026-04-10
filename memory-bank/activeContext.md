# Active Context

## Current Status
The project has undergone a major simplification and optimization phase. The complex automated scheduler has been removed in favor of a powerful, unified CLI tool (`main.py`) that can be run on-demand or via simple OS-level scheduling (cron/Task Scheduler).

## Recent Accomplishments
- **Gemma 4 Migration**: Upgraded cognitive analysis to `gemma4:e4b` for superior reasoning and technical/fundamental synthesis.
- **AlphaEar Integration**: Integrated real-time financial news and social trends via the `alphaear-news` skill, providing better market sentiment context to the LLM.
- **Autonomous Scheduler**: Created `schedule.py`, a robust Windows-compatible scheduler for continuous trading (8:30-18:00, Mon-Fri) with a live monitoring dashboard.
- **Accuracy First Logic**: Refined the `EnhancedDecisionEngine` and `AdvancedRiskManager` to prioritize "justesse" (accuracy) over frequency, implementing stricter consensus and confidence checks.
- **TimesFM 2.5 Integration**: Successfully integrated Google Research's TimesFM 2.5 via a custom automated setup script.
- **Trading 212 Production**: Fully integrated Trading 212 API for real trade execution in both DEMO and LIVE modes.

## Next Steps
- **Demo Mode Monitoring**: Run the system on the T212 Demo account via `schedule.py` to evaluate the "Accuracy First" philosophy.
- **Sentiment Refinement**: Further tune the weight between Alpha Vantage (ticker-specific) and AlphaEar (macro trends) sentiment scores.

## Active Decisions
- **Manual Control**: Shifted from an "automated agent" philosophy to a "decision support tool" philosophy to give the user more control and reduce complexity.
- **Local AI Priority**: Maintained the requirement for local Ollama/Gemma 3 to ensure data privacy and zero cost.
