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
DEFAULT_TICKER = "SXRV_EQ"   # Ticker T212 NASDAQ (iShares)
# Mapping Ticker Yahoo -> Ticker T212
TICKER_MAPPING_T212 = {
    "SXRV.DE": "SXRVd_EQ",
    "SXRV.FRK": "SXRVd_EQ",
    "CRUDP.PA": "CRUDl_EQ",
    "CRUDP": "CRUDl_EQ"
}

def get_auth_header():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return {"Authorization": f"Basic {base64_auth}"}

def load_portfolio_state(ticker=None):
    if not os.path.exists(STATE_FILE):
        # État initial par défaut
        initial = {
            "tickers": {}
        }
        return initial
    
    with open(STATE_FILE, 'r') as f:
        try:
            state = json.load(f)
        except json.JSONDecodeError:
            state = {"tickers": {}}
            
    # Migration si c'est l'ancien format (format plat)
    if "current_capital" in state and "tickers" not in state:
        old_ticker = state.get("active_position", {}).get("ticker", DEFAULT_TICKER) if state.get("active_position") else DEFAULT_TICKER
        state = {
            "tickers": {
                old_ticker: state
            }
        }
        # On sauvegarde immédiatement le nouveau format
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    if ticker:
        # Nettoyage du ticker pour la clé
        clean_ticker = ticker.split('.')[0]
        if clean_ticker not in state["tickers"]:
            state["tickers"][clean_ticker] = {
                "initial_budget": 1000.0, 
                "current_capital": 1000.0, 
                "total_realized_pl": 0.0, 
                "active_position": None
            }
        return state["tickers"][clean_ticker]
    
    return state

def save_portfolio_state(ticker_state, ticker):
    # Nettoyage du ticker pour la clé
    clean_ticker = ticker.split('.')[0]
    
    # Charger l'état complet
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            try:
                full_state = json.load(f)
            except:
                full_state = {"tickers": {}}
    else:
        full_state = {"tickers": {}}
    
    # S'assurer que la structure est correcte
    if "tickers" not in full_state:
        full_state = {"tickers": {}}

    # Mettre à jour le ticker spécifique
    ticker_state["last_update"] = datetime.datetime.now().isoformat()
    full_state["tickers"][clean_ticker] = ticker_state
    
    with open(STATE_FILE, 'w') as f:
        json.dump(full_state, f, indent=4)

def get_real_price_eur(ticker_yahoo=None):
    """Récupère le prix le plus frais possible via Alpha Vantage."""
    target = ticker_yahoo if ticker_yahoo else "SXRV.DE"
    if MarketDataManager:
        try:
            # S'assurer que target est une chaîne de caractères
            if isinstance(target, list) or isinstance(target, tuple):
                target = target[0]
            
            dm = MarketDataManager(target)
            df = dm.get_price_data(force_refresh=True)
            if not df.empty:
                return float(df['close'].iloc[-1])
        except Exception as e:
            print(f"⚠️ Erreur récupération prix Alpha Vantage ({target}) : {e}")
    
    # Fallback estimations
    if "SXRV" in target: return 1240.0
    if "CRUD" in target: return 12.50
    return 100.0 

def execute_t212_trade(signal, confidence, ticker=DEFAULT_TICKER):
    # Mapping du ticker Yahoo vers le ticker T212
    t212_ticker = TICKER_MAPPING_T212.get(ticker, ticker.split('.')[0])
    
    # Charger l'état spécifique au ticker (on utilise le ticker T212 comme clé)
    state = load_portfolio_state(t212_ticker)
    
    env = os.getenv("T212_ENV", "demo").lower()
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = get_auth_header()

    if signal not in ["BUY", "SELL"]:
        return

    def safe_request(method, url, **kwargs):
        for attempt in range(3):
            resp = requests.request(method, url, **kwargs)
            if resp.status_code == 429 or (resp.status_code == 400 and "TooManyRequests" in resp.text):
                wait = (attempt + 1) * 2
                print(f"⚠️ Rate limit atteint, attente de {wait}s...")
                time.sleep(wait)
                continue
            return resp
        return resp

    def get_portfolio_info():
        """Vérifie le cash et les positions réelles sur Trading 212."""
        summary = safe_request("GET", f"{base_url}/equity/account/summary", headers=headers)
        positions = safe_request("GET", f"{base_url}/equity/positions", headers=headers)
        
        info = {"cash": 0.0, "positions": []}
        if summary.status_code == 200:
            info["cash"] = summary.json().get("cash", {}).get("availableToTrade", 0.0)
        if positions.status_code == 200:
            info["positions"] = positions.json()
        return info

    print(f"\n--- 🤖 EXÉCUTION IA TRADING 212 ({env.upper()}) POUR {t212_ticker} ---")
    
    # Vérification systématique avant action
    portfolio = get_portfolio_info()
    print(f"📊 VÉRIFICATION PORTEFEUILLE RÉEL :")
    print(f"   - Cash total disponible : {portfolio['cash']:.2f} €")
    
    # Trouver la position spécifique si elle existe
    current_pos = next((p for p in portfolio['positions'] if p['instrument']['ticker'] == t212_ticker), None)
    
    if current_pos:
        print(f"   - Position détectée : {current_pos['quantity']} actions de {t212_ticker}")
    else:
        print(f"   - Aucune position ouverte sur {t212_ticker}")

    if signal == "BUY":
        if state.get("active_position"):
            print(f"⚠️ Position déjà active pour {t212_ticker} dans le suivi.")
            return

        # 1. Obtenir le prix le plus précis possible
        current_price = get_real_price_eur(ticker)
        print(f"🔍 CALCUL DU PRIX DU MARCHÉ : {current_price} € / action")

        # 2. Calculer la quantité
        available_cash = state.get("current_capital", 1000.0)
        # On vérifie si on a assez de cash réel sur le compte T212
        if portfolio['cash'] < available_cash:
            print(f"⚠️ Pas assez de cash réel ({portfolio['cash']:.2f}€) pour le budget cible ({available_cash:.2f}€).")
            # Option : Ajuster au cash réel disponible ?
            # available_cash = portfolio['cash']
        
        target_budget = available_cash * 0.999 
        quantity = round(target_budget / current_price, 4) 
        
        estimated_cost = quantity * current_price
        print(f"📊 CALCUL QUANTITÉ FRACTIONNÉE :")
        print(f"   - Budget cible : {available_cash:.2f} €")
        print(f"   - Quantité calculée : {quantity} actions")
        print(f"   - Coût estimé : {estimated_cost:.2f} €")

        if quantity <= 0:
            print("❌ Quantité nulle ou négative, abandon.")
            return

        # 3. Passage de l'ordre
        print(f"🚀 Envoi de l'ordre d'achat de {quantity} {t212_ticker}...")
        order_data = {"ticker": t212_ticker, "quantity": quantity}
        resp = safe_request("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)
        
        if resp.status_code in [200, 201, 202]:
            print(f"✅ Ordre placé ! Quantité : {quantity}")
            state["active_position"] = {
                "ticker": t212_ticker,
                "quantity": quantity,
                "buy_budget": estimated_cost,
                "entry_time": datetime.datetime.now().isoformat()
            }
            save_portfolio_state(state, t212_ticker)
        else:
            print(f"❌ Échec de l'achat : {resp.text}")

    elif signal == "SELL":
        if not state.get("active_position") and not current_pos:
            print(f"⚠️ Pas de position active pour {t212_ticker}.")
            return
            
        if not current_pos:
            print(f"⚠️ Position présente dans le suivi mais INTROUVABLE sur T212. Reset du suivi.")
            state["active_position"] = None
            save_portfolio_state(state, t212_ticker)
            return

        # Vente de TOUTE la quantité possédée sur T212
        total_qty = current_pos["quantityAvailableForTrading"]
        current_value_eur = current_pos["walletImpact"]["currentValue"]
        
        print(f"📉 Vente de TOUTE la position sur {t212_ticker} ({total_qty} actions)")
        
        order_data = {"ticker": t212_ticker, "quantity": -total_qty}
        sell_resp = safe_request("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)
        
        if sell_resp.status_code in [200, 201, 202]:
            print(f"✅ Vente effectuée.")
            # Calcul du profit/perte
            buy_cost = state["active_position"]["buy_budget"] if state.get("active_position") else current_value_eur
            state["current_capital"] = current_value_eur
            state["total_realized_pl"] += (current_value_eur - buy_cost)
            state["active_position"] = None
            save_portfolio_state(state, t212_ticker)
            print(f"💰 Nouveau capital pour {t212_ticker} : {state['current_capital']:.2f} €")
        else:
            print(f"❌ Erreur lors de la vente : {sell_resp.text}")

if __name__ == "__main__":
    print("Exécuteur corrigé.")
