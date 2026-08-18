import shutil
import subprocess
import sys
from pathlib import Path

def test_morning_brief_dir_creation():
    output_dir = Path("morning_brief/output")
    # Simulate fresh repo without output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    assert not output_dir.exists()

    # Importing or running morning_brief should create output and not raise FileNotFoundError
    res = subprocess.run(
        [sys.executable, "-c", "import morning_brief.morning_brief as mb; print('Import OK')"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
    assert output_dir.exists(), "output_dir was not created"
    assert (output_dir / "tools").exists(), "output/tools was not created"
    assert (output_dir / "morning_brief.log").exists(), "morning_brief.log was not created"
    print("Test Morning Brief Init: SUCCESS")

if __name__ == "__main__":
    test_morning_brief_dir_creation()
