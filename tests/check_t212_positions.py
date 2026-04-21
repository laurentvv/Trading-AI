import os
import requests
from dotenv import load_dotenv


def check_positions():
    load_dotenv(".env.t212")
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    env = os.getenv("T212_ENV", "demo").lower()

    if not api_key or not api_secret:
        print("Erreur: Clés API manquantes.")
        return

    import base64

    auth_str = f"{api_key}:{api_secret}"
    base64_auth = base64.b64encode(auth_str.encode("ascii")).decode("ascii")
    headers = {"Authorization": f"Basic {base64_auth}"}

    base_url = f"https://{env}.trading212.com/api/v0"

    print(f"--- 📊 VÉRIFICATION PORTEFEUILLE RÉEL ({env.upper()}) ---")
    resp = requests.get(f"{base_url}/equity/positions", headers=headers)

    if resp.status_code == 200:
        positions = resp.json()
        if not positions:
            print("Aucune position ouverte.")
        for p in positions:
            ticker = p.get("instrument", {}).get("ticker", "N/A")
            qty = p.get("quantity", 0)
            val = p.get("walletImpact", {}).get("currentValue", 0)
            profit = p.get("ppl", 0)  # Profit/Perte non-réalisé
            print(f"✅ Instrument: {ticker}")
            print(f"   - Quantité: {qty}")
            print(f"   - Valeur actuelle: {val:.2f} €")
            print(f"   - Profit/Perte (PPL): {profit:+.2f} €")
    else:
        print(f"Erreur API: {resp.status_code} - {resp.text}")


if __name__ == "__main__":
    check_positions()
