import shutil
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_morning_brief_dir_creation(tmp_path):
    """L'import de morning_brief crée `output/` et `output/tools` s'ils manquent.

    Le test travaille sur une COPIE du package (sans `output/`) : il ne doit jamais supprimer
    le `morning_brief/output/` réel du dépôt (brief du jour lu par le contexte LLM).
    """
    pkg_copy = tmp_path / "morning_brief"
    shutil.copytree(
        REPO_ROOT / "morning_brief",
        pkg_copy,
        ignore=shutil.ignore_patterns("output", "__pycache__"),
    )
    output_dir = pkg_copy / "output"
    # Simulate fresh repo without output directory
    assert not output_dir.exists()

    # `cwd=tmp_path` → le package importé est la copie ; le dépôt reste accessible pour les imports `src`.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT)])}
    res = subprocess.run(
        [sys.executable, "-c", "import morning_brief.morning_brief as mb; print('Import OK')"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
    assert output_dir.exists(), "output_dir was not created"
    assert (output_dir / "tools").exists(), "output/tools was not created"
    assert (output_dir / "morning_brief.log").exists(), "morning_brief.log was not created"
    print("Test Morning Brief Init: SUCCESS")
