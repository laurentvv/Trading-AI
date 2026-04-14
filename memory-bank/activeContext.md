# Active Context

## Current Status
The project is now in a **high-fidelity production/demo phase**. The decision engine has been refined for extreme accuracy, and the web research capabilities have been significantly upgraded. The system is operating under an "Accuracy First" (Justesse) mandate.

### Key Recent Changes
- **Vincent Ganne Model Refinement**: Now explicitly exclusive to **Nasdaq** assets for market bottom validation. It only generates `BUY` signals and acts as a geopolitical safety lock (blocking Nasdaq buys if energy prices are > $94). It is disabled for Oil trading to avoid self-referential bias.
- **Crawl4AI Integration**: Replaced simple DuckDuckGo snippets with full-page asynchronous crawling for macro research, providing the LLM with dense, high-quality context.
- **Dynamic Prompt Engineering**: LLM prompts are now ticker-aware, include qualified indicators (e.g., RSI qualificators like 'Overbought'), and incorporate current temporal context (Month/Year) and 5-day price trends for search query generation.
- **Trading 212 Dashboard**: Fixed display bugs and improved the detailed logging in `trading_journal.csv` to track each individual model's contribution (Classic, LLM, TimesFM, VG, Sentiment).
- **Project Cleanup**: Root directory has been cleaned; all test scripts moved to `tests/`.

## Immediate Objectives
- [x] Integrate Vincent Ganne geopolitical safety rules.
- [x] Implement ticker-specific LLM prompts.
- [x] Enable high-fidelity web crawling with Crawl4AI.
- [x] Validate full cycle on Nasdaq and Oil tickers.
- [x] Implement detailed per-model logging.
- [ ] Monitor real-time performance in Demo Mode.
- [ ] Add automated Stop-Loss rules in AdvancedRiskManager.

## Decision Log
- **Nasdaq Exclusivity for VG**: Decided to restrict the Vincent Ganne model to Nasdaq because its energy-price-to-stock-bottom logic is fundamentally a cross-asset indicator for equities, not a directional signal for energy itself.
- **BUY-Only Signal for VG**: Restricted VG model to BUY signals to prevent it from forcing exits on Nasdaq when energy prices spike, which might be noise rather than a structural top.
- **Dynamic Search Queries**: Improved query generation by adding the current date to search queries, ensuring the LLM finds recent news instead of outdated macro reports.
