with open("src/enhanced_decision_engine.py", "r") as f:
    content = f.read()

# 1. Update base_weights
if '"tensortrade": 0.10' not in content:
    # We replace the whole dictionary block to adjust weights smoothly to sum to 1.0
    old_weights_block = """        self.base_weights = base_weights or {
            "classic": 0.10,
            "llm_text": 0.20,
            "llm_visual": 0.10,
            "sentiment": 0.10,
            "timesfm": 0.25,
            "vincent_ganne": 0.15,
            "oil_bench": 0.10,
        }"""

    new_weights_block = """        self.base_weights = base_weights or {
            "classic": 0.10,
            "llm_text": 0.15,
            "llm_visual": 0.10,
            "sentiment": 0.10,
            "timesfm": 0.20,
            "vincent_ganne": 0.15,
            "oil_bench": 0.10,
            "tensortrade": 0.10,
        }"""

    content = content.replace(old_weights_block, new_weights_block)

# 2. Update make_enhanced_decision signature
if "tensortrade_decision: Dict = None," not in content:
    content = content.replace(
        "timesfm_decision: Dict = None,",
        "timesfm_decision: Dict = None,\n        tensortrade_decision: Dict = None,",
    )

# 3. Add tensortrade decision to decisions list
if "if tensortrade_decision:" not in content:
    tt_block = """
        if tensortrade_decision:
            decisions.append(
                ModelDecision(
                    signal=tensortrade_decision.get("signal", "HOLD"),
                    confidence=tensortrade_decision.get("confidence", 0.0),
                    strength=self._normalize_signal(
                        tensortrade_decision.get("signal", "HOLD")
                    ),
                    timestamp=timestamp,
                    model_name="tensortrade",
                    reasoning=tensortrade_decision.get(
                        "analysis", "TensorTrade RL policy output"
                    ),
                )
            )
"""
    # Insert it right after timesfm_decision block
    content = content.replace(
        """        if timesfm_decision:
            decisions.append(
                ModelDecision(
                    signal=timesfm_decision.get("signal", "HOLD"),
                    confidence=timesfm_decision.get("confidence", 0.0),
                    strength=self._normalize_signal(
                        timesfm_decision.get("signal", "HOLD")
                    ),
                    timestamp=timestamp,
                    model_name="timesfm",
                    reasoning=timesfm_decision.get(
                        "analysis", "TimesFM time series forecasting"
                    ),
                )
            )""",
        """        if timesfm_decision:
            decisions.append(
                ModelDecision(
                    signal=timesfm_decision.get("signal", "HOLD"),
                    confidence=timesfm_decision.get("confidence", 0.0),
                    strength=self._normalize_signal(
                        timesfm_decision.get("signal", "HOLD")
                    ),
                    timestamp=timestamp,
                    model_name="timesfm",
                    reasoning=timesfm_decision.get(
                        "analysis", "TimesFM time series forecasting"
                    ),
                )
            )"""
        + tt_block,
    )

with open("src/enhanced_decision_engine.py", "w") as f:
    f.write(content)

print("Updated enhanced_decision_engine.py")
