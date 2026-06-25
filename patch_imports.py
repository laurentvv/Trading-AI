with open('src/llm_client.py', 'r') as f:
    content = f.read()

content = content.replace('''try:
    from free_llm_api_keys import FreeLLMClient
    from free_llm_api_keys.exceptions import FreeLLMError
except ImportError:
    FreeLLMClient = None
    FreeLLMError = Exception

try:
    from src.free_llm_api_keys import FreeLLMClient
    from src.free_llm_api_keys.exceptions import FreeLLMError
except ImportError:
    FreeLLMClient = None
    FreeLLMError = Exception''', '''try:
    import sys
    # Ensure vendor dir is in path for imports
    vendor_path = Path(__file__).parent.parent / "vendor" / "free-llm-api-keys-python" / "src"
    if str(vendor_path) not in sys.path:
        sys.path.insert(0, str(vendor_path))
    from free_llm_api_keys import FreeLLMClient
    from free_llm_api_keys.exceptions import FreeLLMError
except ImportError:
    FreeLLMClient = None
    FreeLLMError = Exception''')

with open('src/llm_client.py', 'w') as f:
    f.write(content)
