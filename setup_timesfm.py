import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(command, description):
    print(f"[*] {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

def main():
    print("=== TimesFM 2.5 Setup Script for Trading AI ===")
    
    project_root = Path(__file__).parent
    vendor_dir = project_root / "vendor"
    timesfm_dir = vendor_dir / "timesfm"
    
    # 1. Nettoyage des anciennes installations
    if timesfm_dir.exists():
        print(f"[*] Removing existing TimesFM source at {timesfm_dir}...")
        shutil.rmtree(timesfm_dir)
    
    if not vendor_dir.exists():
        vendor_dir.mkdir()

    # 2. Clonage du dépôt officiel
    repo_url = "https://github.com/google-research/timesfm.git"
    run_command(f"git clone {repo_url} {timesfm_dir}", "Cloning TimesFM repository")

    # 3. Application du patch __init__.py (le fix crucial pour l'API 2.5)
    patch_path = timesfm_dir / "src" / "timesfm" / "timesfm_2p5" / "__init__.py"
    print("[*] Applying API 2.5 patch (__init__.py)...")
    patch_content = "from . import timesfm_2p5_torch\nfrom . import timesfm_2p5_base\n"
    
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_content)

    # 4. Installation via uv en mode éditable
    # Cela permet d'avoir la lib dans le venv tout en gardant nos patchs locaux
    run_command(f"uv pip install -e {timesfm_dir}", "Installing TimesFM in editable mode via uv")

    print("\n[V] TimesFM 2.5 is now correctly installed and patched!")
    print("[*] You can now run the analysis with 'uv run main.py'")

if __name__ == "__main__":
    main()
