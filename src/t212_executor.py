import os
import json
import base64
import requests
import datetime
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ajouter le chemin pour importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.data import MarketDataManager
except ImportError:
    # Fallback si l'import échoue selon le contexte d'exécution
    MarketDataManager = None

load_dotenv(".env.t212")

STATE_FILE = "t212_portfolio_state.json"
DEFAULT_TICKER = "SXRVd_EQ"  # Ticker T212
AV_TICKER = "SXRV.FRK"       # Ticker Alpha Vantage correspondant (iShares Nasdaq 100 Acc sur Francfort)

def get_auth_header():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return {"Authorization": f"Basic {base64_auth}"}

def load_portfolio_state():
    if not os.path.exists(STATE_FILE):
        return {"initial_budget": 1000.0, "current_capital": 1000.0, "total_realized_pl": 0.0, "active_position": None}
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def save_portfolio_state(state):
    state["last_update"] = datetime.datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def get_real_price_eur():
    """Récupère le prix le plus frais possible via Alpha Vantage."""
    if MarketDataManager:
        try:
            dm = MarketDataManager(AV_TICKER)
            df = dm.get_price_data(force_refresh=True)
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ Erreur récupération prix Alpha Vantage : {e}")
    
    # Fallback via une recherche T212 si possible ou dernière estimation connue
    # Note: En journée, SXRV est autour de 1230-1250€ actuellement
    return 1240.0 

def execute_t212_trade(signal, confidence, ticker=DEFAULT_TICKER):
    state = load_portfolio_state()
    env = os.getenv("T212_ENV", "demo").lower()
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = get_auth_header()

    if signal not in ["BUY", "SELL"]:
        return

    print(f"\n--- 🤖 EXÉCUTION IA TRADING 212 ({env.upper()}) ---")
    
    if signal == "BUY":
        if state.get("active_position"):
            print("⚠️ Position déjà active.")
            return

        # 1. Obtenir le prix le plus précis possible
        current_price = get_real_price_eur()
        print(f"📈 Prix actuel estimé : {current_price} €")

        # 2. Calculer la quantité pour viser EXACTEMENT le budget (avec 0.5% de marge de sécurité)
        target_budget = state["current_capital"] * 0.995 
        quantity = round(target_budget / current_price, 4) 
        
        estimated_cost = quantity * current_price
        print(f"💰 Cible : {target_budget}€ | Quantité calculée : {quantity} actions | Coût estimé : {estimated_cost:.2f}€")

        if quantity <= 0:
            print("❌ Quantité nulle ou négative, abandon.")
            return

        # 3. Passage de l'ordre
        order_data = {"ticker": ticker, "quantity": quantity}
        resp = requests.post(f"{base_url}/equity/orders/market", headers=headers, json=order_data)
        
        if resp.status_code in [200, 201, 202]:
            print(f"✅ Ordre placé ! Quantité : {quantity}")
            state["active_position"] = {
                "ticker": ticker,
                "quantity": quantity,
                "buy_budget": estimated_cost,
                "entry_time": datetime.datetime.now().isoformat()
            }
            save_portfolio_state(state)
        else:
            print(f"❌ Échec de l'achat : {resp.text}")

    elif signal == "SELL":
        if not state.get("active_position"):
            print("⚠️ Pas de position active.")
            return
            
        resp = requests.get(f"{base_url}/equity/positions?ticker={ticker}", headers=headers)
        if resp.status_code == 200:
            positions = resp.json()
            if not positions:
                state["active_position"] = None
                save_portfolio_state(state)
                return
            
            # Vente de TOUTE la quantité possédée (gestion des fractions)
            total_qty = positions[0]["quantityAvailableForTrading"]
            current_value_eur = positions[0]["walletImpact"]["currentValue"]
            
            print(f"📉 Vente de TOUTE la position ({total_qty} actions)")
            
            order_data = {"ticker": ticker, "quantity": -total_qty}
            sell_resp = requests.post(f"{base_url}/equity/orders/market", headers=headers, json=order_data)
            
            if sell_resp.status_code in [200, 201, 202]:
                print(f"✅ Vente effectuée.")
                state["current_capital"] = current_value_eur
                state["total_realized_pl"] += (current_value_eur - state["active_position"]["buy_budget"])
                state["active_position"] = None
                save_portfolio_state(state)
                print(f"💰 Nouveau capital : {state['current_capital']:.2f} €")
        else:
            print(f"❌ Erreur récupération positions : {resp.text}")

if __name__ == "__main__":
    print("Exécuteur corrigé.")
