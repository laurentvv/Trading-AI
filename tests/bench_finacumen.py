import subprocess
import sys
import time
import json
from pathlib import Path

def run_bench(runs=3, ticker="CRUDP.PA"):
    print(f"=== Benchmarking FinAcumen sur {runs} itérations ({ticker}) ===")
    
    script_path = str(Path(__file__).parent.parent / "src" / "finacumen_main.py")
    results = []
    
    for i in range(runs):
        print(f"\n--- Itération {i+1}/{runs} ---")
        start_time = time.time()
        
        try:
            # On lance le script
            process = subprocess.run(
                [sys.executable, script_path, "--ticker", ticker],
                capture_output=True,
                text=True,
                timeout=600
            )
            elapsed = time.time() - start_time
            
            if process.returncode == 0:
                print(f"Succès en {elapsed:.2f}s")
                # On lit le fichier de sortie
                state_file = Path("data_cache/finacumen") / f"finacumen_{ticker}.json"
                if state_file.exists():
                    with open(state_file, "r") as f:
                        data = json.load(f)
                        data["time_s"] = elapsed
                        results.append(data)
                        print(f"Signal: {data.get('signal')} | Conf: {data.get('confidence')} | Status: {data.get('status')}")
                else:
                    print("Fichier de sortie introuvable.")
                    results.append({"status": "error_no_file", "time_s": elapsed})
            else:
                print(f"Échec en {elapsed:.2f}s (Code {process.returncode})")
                print("Erreur:", process.stderr[:200])
                results.append({"status": "failed", "time_s": elapsed, "error": process.stderr[:200]})
                
        except subprocess.TimeoutExpired:
            print("Timeout (300s)")
            results.append({"status": "timeout", "time_s": 300})
            
    # Summary
    print("\n" + "="*50)
    print("RÉSUMÉ DU BENCHMARK FINACUMEN")
    print("="*50)
    
    successes = [r for r in results if r.get("status") == "success"]
    times = [r["time_s"] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    
    print(f"Succès: {len(successes)} / {runs}")
    print(f"Temps moyen: {avg_time:.2f}s")
    
    if successes:
        signals = [r.get("signal") for r in successes]
        from collections import Counter
        counts = Counter(signals)
        print(f"Répartition des signaux: {dict(counts)}")
        
        confs = [r.get("confidence", 0) for r in successes]
        avg_conf = sum(confs) / len(confs)
        print(f"Confiance moyenne: {avg_conf:.2f}")

if __name__ == "__main__":
    run_bench()
