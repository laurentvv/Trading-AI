import os
import base64
import requests
import json
import time
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv(".env.t212")

def get_auth_header(api_key, api_secret):
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return f"Basic {base64_auth}"

def audit_api():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    env = os.getenv("T212_ENV", "demo").lower()
    
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = {
        "Authorization": get_auth_header(api_key, api_secret),
        "Content-Type": "application/json"
    }

    tests = [
        ("Résumé du compte", "/equity/account/summary"),
        ("Positions ouvertes", "/equity/positions"),
        ("Ordres en attente", "/equity/orders"),
        ("Métadonnées : Bourses", "/equity/metadata/exchanges"),
        ("Historique : Ordres (limit 1)", "/equity/history/orders?limit=1"),
        ("Historique : Dividendes (limit 1)", "/equity/history/dividends?limit=1"),
        ("Historique : Transactions (limit 1)", "/equity/history/transactions?limit=1"),
        ("Métadonnées : Instruments", "/equity/metadata/instruments"),
    ]

    print(f"🚀 Démarrage de l'audit complet de l'API (Environnement: {env.upper()})\n")
    print(f"{'Endpoint':<40} | {'Status':<10} | {'Détails'}")
    print("-" * 80)

    for label, endpoint in tests:
        try:
            # Pour l'historique et les instruments, on ne veut pas surcharger le log
            response = requests.get(f"{base_url}{endpoint}", headers=headers)
            
            status_icon = "✅ OK" if response.status_code == 200 else f"❌ {response.status_code}"
            
            detail = ""
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    detail = f"{len(data)} éléments trouvés"
                elif isinstance(data, dict):
                    if "items" in data:
                        detail = f"{len(data['items'])} éléments dans l'historique"
                    else:
                        detail = "Données reçues"
            elif response.status_code == 403:
                detail = "Permission manquante (403 Forbidden)"
            elif response.status_code == 401:
                detail = "Erreur d'authentification"
            else:
                detail = response.text[:50] + "..." if len(response.text) > 50 else response.text

            print(f"{label:<40} | {status_icon:<10} | {detail}")
            
            # Respecter les limites de taux (rate limits)
            time.sleep(1) 

        except Exception as e:
            print(f"{label:<40} | 💥 ERROR    | {str(e)}")

    print("\n💡 Note : Pour passer un ordre d'achat/vente (POST), un test spécifique est requis.")

if __name__ == "__main__":
    audit_api()
