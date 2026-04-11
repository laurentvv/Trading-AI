🧪 Add tests for sentiment analysis logic

🎯 **What:**
Added unit tests for `get_sentiment_decision_from_score` in `src/sentiment_analysis.py` to fill a missing testing gap.

📊 **Coverage:**
- BUY signals for scores > 0.15
- SELL signals for scores < -0.15
- HOLD signals for scores between -0.15 and 0.15
- Exact boundary checks (0.15 and -0.15 resolve to HOLD)
- Confidence capping behavior (confidence does not exceed 1.0)
- Analysis string formatting

✨ **Result:**
Enhanced the test suite with robust coverage of sentiment analysis decisions, creating a safety net for future refactoring and capturing exact expected threshold logic.
