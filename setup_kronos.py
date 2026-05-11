import os
import subprocess
import sys
import shutil
import stat
from pathlib import Path


def on_rm_error(func, path, exc_info):
    """
    Handler for shutil.rmtree error on Windows (Read-only files).
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def run_command(command, description):
    print(f"[*] {description}...")
    try:
        if isinstance(command, str):
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error: {e}")
        sys.exit(1)


def main():
    print("=== Kronos Setup Script for Trading AI ===")

    project_root = Path(__file__).parent
    vendor_dir = project_root / "vendor"
    kronos_dir = vendor_dir / "kronos"

    # 1. Clean up existing installations
    if kronos_dir.exists():
        print(f"[*] Removing existing Kronos source at {kronos_dir}...")
        try:
            shutil.rmtree(kronos_dir, onerror=on_rm_error)
        except Exception as e:
            print(f"[!] Warning: Could not remove directory perfectly: {e}")
            old_dir = f"{kronos_dir}_old_{int(os.path.getmtime(kronos_dir))}"
            try:
                os.rename(kronos_dir, old_dir)
                print(f"[*] Renamed old directory to {old_dir}")
            except Exception:
                print(f"[!] CRITICAL: Path {kronos_dir} is locked.")
                sys.exit(1)

    if not vendor_dir.exists():
        vendor_dir.mkdir()

    # 2. Clone the official repository
    repo_url = "https://github.com/shiyu-coder/Kronos.git"
    run_command(f"git clone {repo_url} {kronos_dir}", "Cloning Kronos repository")

    # 3. Check for requirements and install via uv
    req_file = kronos_dir / "requirements.txt"
    if req_file.exists():
        run_command(f"uv pip install -r {req_file}", "Installing Kronos dependencies")
    else:
        print("[*] No requirements.txt found, skipping dependency installation.")

    print("\n[V] Kronos is now correctly installed!")
    print("[*] You can now run the analysis with 'uv run main.py'")


if __name__ == "__main__":
    main()
