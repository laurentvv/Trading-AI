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
    return f"Basic {base64_auth}"

def test_connection():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    env = os.getenv("T212_ENV", "demo").lower()
    
    if not api_key or not api_secret:
        print("❌ Erreur : T212_API_KEY ou T212_API_SECRET non trouvés dans l'environnement.")
        print("Assurez-vous d'avoir créé un fichier .env.t212 avec vos identifiants.")
        return

    base_url = f"https://{env}.trading212.com/api/v0"
    headers = {
        "Authorization": get_auth_header(api_key, api_secret),
        "Content-Type": "application/json"
    }

    print(f"--- Test de connexion (Environnement: {env.upper()}) ---")
    
    # 1. Résumé du compte
    print("\n1. Récupération du résumé du compte...")
    try:
        response = requests.get(f"{base_url}/equity/account/summary", headers=headers)
        if response.status_code == 200:
            print("✅ Succès !")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Échec (Code: {response.status_code})")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Erreur lors de l'appel : {str(e)}")
        return

    # 2. Consultation des positions
    print("\n2. Récupération des positions...")
    try:
        response = requests.get(f"{base_url}/equity/positions", headers=headers)
        if response.status_code == 200:
            positions = response.json()
            print(f"✅ Succès ! ({len(positions)} position(s) trouvée(s))")
            if positions:
                for pos in positions:
                    instr = pos.get('instrument', {})
                    print(f"  - {instr.get('name')} ({instr.get('ticker')}): {pos.get('quantity')} actions")
            else:
                print("  Aucune position ouverte.")
        else:
            print(f"❌ Échec (Code: {response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"❌ Erreur lors de l'appel : {str(e)}")

if __name__ == "__main__":
    test_connection()
