# Fix src/data.py
with open("src/data.py", "r") as f:
    content = f.read()

# Check if numpy is imported at top
if "import numpy as np" not in content:
    content = "import numpy as np\n" + content

with open("src/data.py", "w") as f:
    f.write(content)

# Fix src/tensortrade_model.py
with open("src/tensortrade_model.py", "r") as f:
    content = f.read()

content = content.replace(
    "if action == 1: reward = price_change",
    "if action == 1:\n                        reward = price_change",
)
content = content.replace(
    "elif action == 2: reward = -price_change",
    "elif action == 2:\n                        reward = -price_change",
)
content = content.replace("if done: break", "if done:\n                break")
content = content.replace(
    'if action == 1: signal = "BUY"', 'if action == 1:\n            signal = "BUY"'
)
content = content.replace(
    'elif action == 2: signal = "SELL"',
    'elif action == 2:\n            signal = "SELL"',
)

with open("src/tensortrade_model.py", "w") as f:
    f.write(content)

# Fix tests/test_llm_prompts.py
with open("tests/test_llm_prompts.py", "r") as f:
    content = f.read()

content = content.replace(
    "if dummy_path.exists(): dummy_path.unlink()",
    "if dummy_path.exists():\n            dummy_path.unlink()",
)

with open("tests/test_llm_prompts.py", "w") as f:
    f.write(content)
