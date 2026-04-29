with open("src/enhanced_trading_example.py", "r") as f:
    content = f.read()

# 1. Add import
if "from tensortrade_model import get_tensortrade_prediction" not in content:
    content = content.replace(
        "from timesfm_model import get_timesfm_prediction",
        "from timesfm_model import get_timesfm_prediction\nfrom tensortrade_model import get_tensortrade_prediction",
    )

# 2. Add tensortrade to get_model_predictions
if "tensortrade_decision = get_tensortrade_prediction(" not in content:
    content = content.replace(
        "timesfm_decision = get_timesfm_prediction(data_with_features)",
        "timesfm_decision = get_timesfm_prediction(data_with_features)\n        tensortrade_decision = get_tensortrade_prediction(data_with_features)",
    )

    content = content.replace(
        '"timesfm": timesfm_decision,',
        '"timesfm": timesfm_decision,\n            "tensortrade": tensortrade_decision,',
    )

# 3. Add tensortrade to perform_enhanced_analysis
if 'tensortrade_decision=model_predictions["tensortrade"]' not in content:
    content = content.replace(
        'timesfm_decision=model_predictions["timesfm"],',
        'timesfm_decision=model_predictions["timesfm"],\n            tensortrade_decision=model_predictions["tensortrade"],',
    )

with open("src/enhanced_trading_example.py", "w") as f:
    f.write(content)

print("Updated enhanced_trading_example.py")
