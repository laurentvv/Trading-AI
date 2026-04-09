import os
import base64
import requests
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv(".env.t212")

def get_auth_header(api_key, api_secret):
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return {"Authorization": f"Basic {base64_auth}"}

def find_nasdaq_etf_eur():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    env = os.getenv("T212_ENV", "demo").lower()
    
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = get_auth_header(api_key, api_secret)

    print(f"🔍 Recherche d'ETFs Nasdaq en EUR sur {env.upper()}...")
    
    try:
        # Récupération de tous les instruments (Note: cet appel est limité à 1/50s)
        response = requests.get(f"{base_url}/equity/metadata/instruments", headers=headers)
        
        if response.status_code == 200:
            instruments = response.json()
            results = []
            
            for inst in instruments:
                name = inst.get("name", "").upper()
                ticker = inst.get("ticker", "").upper()
                currency = inst.get("currencyCode", "").upper()
                inst_type = inst.get("type", "").upper()
                
                # Filtres : Nasdaq + ETF + EUR
                if "NASDAQ" in name and inst_type == "ETF" and currency == "EUR":
                    results.append({
                        "ticker": inst.get("ticker"),
                        "name": inst.get("name"),
                        "shortName": inst.get("shortName"),
                        "isin": inst.get("isin"),
                        "currency": currency
                    })
            
            if results:
                print(f"\n✅ {len(results)} ETF(s) trouvé(s) :\n")
                print(f"{'Ticker T212':<15} | {'Nom':<40} | {'ISIN'}")
                print("-" * 75)
                for res in results:
                    print(f"{res['ticker']:<15} | {res['name'][:40]:<40} | {res['isin']}")
            else:
                print("\n❌ Aucun ETF Nasdaq trouvé en EUR.")
        else:
            print(f"\n❌ Erreur API ({response.status_code}) : {response.text}")
            
    except Exception as e:
        print(f"\n💥 Erreur : {str(e)}")

if __name__ == "__main__":
    find_nasdaq_etf_eur()
