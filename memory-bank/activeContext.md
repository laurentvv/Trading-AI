# Active Context

## 1. Current Work Focus
The project is in **Phase 3: Finalization and Documentation**.
The immediate focus is on **Step 5: Update the Memory Bank** and **Step 6: Update the `README.md`**.

## 2. Recent Changes
- **3-Model Hybrid Engine Integrated**: `src/main.py` has been successfully updated to orchestrate the classic model, the text LLM, and the new visual LLM, combining their outputs into a final decision.
- **LLM Client Upgraded**: The `src/llm_client.py` module supports both text and visual models.
- **Chart Generator Created**: The `src/chart_generator.py` module is complete.
- **Bug Fix (2025-08-18)**: Fixed a crash caused by a missing `ALPHA_VANTAGE_API_KEY`. The system now loads the key from a `.env` file.

## 3. Next Steps
1.  Update all Memory Bank files to reflect the final, 3-model architecture.
2.  Update the main `README.md` to include instructions and information about the new visual AI feature.
3.  Perform a final end-to-end test of the complete system.
4.  Submit the final project.

## 4. Active Decisions & Considerations
- Charts will be generated internally using `mplfinance` for reliability.
- The chart will show 6 months of daily data with candlesticks, 50/200 MAs, Volume, RSI, and MACD.
- The new visual signal will be a third, equal vote in the hybrid engine.
