with open("tests/test_except_fix.py", "r") as f:
    content = f.read()

# Fix docstring order
old_test = """@patch("src.t212_executor.sync_state_from_t212")
def test_except_handling(mock_sync):
    mock_sync.side_effect = Exception("Mocked T212 error")
    \"\"\"Test that corrupted state file is handled gracefully.\"\"\""""

new_test = """@patch("src.t212_executor.sync_state_from_t212")
def test_except_handling(mock_sync):
    \"\"\"Test that corrupted state file is handled gracefully.\"\"\"
    mock_sync.side_effect = Exception("Mocked T212 error")"""

if old_test in content:
    content = content.replace(old_test, new_test)
else:
    print("Could not find the test method to fix")

with open("tests/test_except_fix.py", "w") as f:
    f.write(content)
