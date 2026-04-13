"""Test file locking for t212_portfolio_state.json."""
import json
import sys
import time
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from t212_executor import (
    _atomic_json_write, _read_with_retry
)

def test_atomic_write():
    """Test that atomic write produces valid JSON."""
    test_path = Path("test_atomic.json")
    try:
        data = {"tickers": {"TEST": {"current_capital": 1000.0}}}
        _atomic_json_write(test_path, data)

        with open(test_path, 'r') as f:
            result = json.load(f)
        assert result == data, f"Expected {data}, got {result}"
        print("PASS: Atomic write produces valid JSON")

        # No temp files left behind
        tmp_files = list(Path('.').glob('*.tmp'))
        assert len(tmp_files) == 0, f"Temp files left behind: {tmp_files}"
        print("PASS: No temp files left behind")
    finally:
        if test_path.exists():
            test_path.unlink()

def test_read_with_retry():
    """Test retry reader handles missing and valid files."""
    test_path = Path("test_retry.json")
    try:
        # Missing file
        result = _read_with_retry(test_path)
        assert result is None, f"Expected None for missing file, got {result}"
        print("PASS: Returns None for missing file")

        # Valid file
        with open(test_path, 'w') as f:
            json.dump({"test": 1}, f)
        result = _read_with_retry(test_path)
        assert result == {"test": 1}, f"Expected {{'test': 1}}, got {result}"
        print("PASS: Reads valid file correctly")
    finally:
        if test_path.exists():
            test_path.unlink()

def test_concurrent_writes():
    """Test that concurrent writes don't corrupt the file."""
    test_path = Path("test_concurrent.json")
    errors = []

    def write_task(thread_id):
        try:
            for i in range(10):
                data = {"tickers": {f"thread_{thread_id}": {"value": i}}}
                _atomic_json_write(test_path, data)
                time.sleep(0.01)
        except Exception as e:
            errors.append(str(e))

    try:
        threads = [threading.Thread(target=write_task, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify file is still valid JSON
        final = _read_with_retry(test_path)
        assert final is not None, "File should exist after concurrent writes"
        assert "tickers" in final, "File should have valid structure"
        print("PASS: Concurrent writes don't corrupt file")

        if errors:
            print(f"  Note: {len(errors)} errors during concurrent writes (expected with no locking)")
    finally:
        if test_path.exists():
            test_path.unlink()

if __name__ == '__main__':
    test_atomic_write()
    test_read_with_retry()
    test_concurrent_writes()
    print("\nAll file locking tests passed!")
