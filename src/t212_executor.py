import os
import json
import base64
import logging
import requests
import datetime
import time
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# File locking constants
STATE_LOCK_TIMEOUT = 5  # seconds to wait for lock
STATE_LOCK_RETRIES = 3
STATE_LOCK_RETRY_DELAY = 0.5  # seconds between retries


def _atomic_json_write(filepath: Path, data: dict):
    """
    Atomically write JSON data using temp file + rename pattern.
    This prevents corruption if two processes write simultaneously.
    On both Windows and POSIX, os.replace() is atomic.
    """
    dir_path = filepath.parent
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(dir_path))
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(data, tmp_file, indent=4)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        # Atomic rename (os.replace is atomic on both Windows and POSIX)
        os.replace(tmp_path, str(filepath))
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_with_retry(filepath: Path, max_retries: int = STATE_LOCK_RETRIES):
    """
    Read JSON file with retry for robustness against concurrent writes.
    """
    for attempt in range(max_retries):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # File might be in the middle of being written, retry
            if attempt < max_retries - 1:
                time.sleep(STATE_LOCK_RETRY_DELAY)
                continue
            raise
        except FileNotFoundError:
            return None
    return None


# Ajouter le chemin pour importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.data import MarketDataManager
    from src.database import insert_transaction, insert_portfolio_state
except ImportError:
    # Fallback si l'import échoue selon le contexte d'exécution
    MarketDataManager = None
    insert_transaction = None
    insert_portfolio_state = None

load_dotenv(".env.t212")

STATE_FILE = "t212_portfolio_state.json"
DEFAULT_TICKER = "SXRV_EQ"  # Ticker T212 NASDAQ (iShares)
# Mapping Ticker Yahoo -> Ticker T212
TICKER_MAPPING_T212 = {
    "SXRV.DE": "SXRVd_EQ",
    "SXRV.FRK": "SXRVd_EQ",
    "CRUDP.PA": "CRUDl_EQ",
    "CRUDP": "CRUDl_EQ",
}


def get_t212_ticker(ticker_yahoo: str) -> str:
    """Consistently maps a Yahoo ticker to a T212 instrument ticker."""
    if not ticker_yahoo:
        return DEFAULT_TICKER
    # Use mapping if available, otherwise use prefix
    return TICKER_MAPPING_T212.get(ticker_yahoo, ticker_yahoo.split(".")[0])


def get_auth_header():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "T212_API_KEY or T212_API_SECRET is missing. Please set it in your environment or .env.t212 file."
        )
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return {"Authorization": f"Basic {base64_auth}"}


def load_portfolio_state(ticker=None):
    if not os.path.exists(STATE_FILE):
        # État initial par défaut
        initial = {"tickers": {}}
        return initial

    state = _read_with_retry(Path(STATE_FILE))
    if state is None:
        state = {"tickers": {}}

    # Migration si c'est l'ancien format (format plat)
    if "current_capital" in state and "tickers" not in state:
        old_ticker = (
            state.get("active_position", {}).get("ticker", DEFAULT_TICKER)
            if state.get("active_position")
            else DEFAULT_TICKER
        )
        state = {"tickers": {old_ticker: state}}
        # On sauvegarde immédiatement le nouveau format
        _atomic_json_write(Path(STATE_FILE), state)

    if ticker:
        # Nettoyage du ticker pour la clé via le helper standard
        clean_ticker = get_t212_ticker(ticker)
        if clean_ticker not in state["tickers"]:
            state["tickers"][clean_ticker] = {
                "initial_budget": 1000.0,
                "current_capital": 1000.0,
                "total_realized_pl": 0.0,
                "active_position": None,
            }

        # Nettoyage récursif si la migration a foiré (évite le "tickers" dans "tickers")
        if "tickers" in state["tickers"][clean_ticker]:
            del state["tickers"][clean_ticker]["tickers"]

        return state["tickers"][clean_ticker]

    return state


def save_portfolio_state(ticker_state, ticker):
    # Nettoyage du ticker pour la clé via le helper standard
    clean_ticker = get_t212_ticker(ticker)

    # Charger l'état complet avec retry
    full_state = _read_with_retry(Path(STATE_FILE))
    if full_state is None:
        full_state = {"tickers": {}}

    # S'assurer que la structure est correcte
    if "tickers" not in full_state:
        full_state = {"tickers": {}}

    # Nettoyage de sécurité avant sauvegarde
    if "tickers" in ticker_state:
        del ticker_state["tickers"]

    # Mettre à jour le ticker spécifique
    ticker_state["last_update"] = datetime.datetime.now().isoformat()
    full_state["tickers"][clean_ticker] = ticker_state

    # Atomic write to prevent corruption
    _atomic_json_write(Path(STATE_FILE), full_state)


def get_t212_price(ticker_yahoo: str) -> float | None:
    """Fetch live price from T212 via /equity/positions (only works for open positions)."""
    t212_ticker = get_t212_ticker(ticker_yahoo)
    try:
        env = os.getenv("T212_ENV", "demo").lower()
        base_url = f"https://{env}.trading212.com/api/v0"
        headers = get_auth_header()
        resp = requests.get(f"{base_url}/equity/positions", headers=headers, timeout=5)
        if resp.status_code == 200:
            for pos in resp.json():
                if pos["instrument"]["ticker"] == t212_ticker:
                    price = float(pos["currentPrice"])
                    logger.info(
                        f"T212 live price for {ticker_yahoo} ({t212_ticker}): {price:.2f} EUR"
                    )
                    return price
            logger.debug(f"No T212 position found for {t212_ticker}, price unavailable")
    except Exception as e:
        logger.debug(f"T212 price fetch failed for {ticker_yahoo}: {e}")
    return None


def get_real_price_eur(ticker_yahoo=None):
    """Best-effort price retrieval: T212 live → MarketDataManager → yfinance history."""
    target = ticker_yahoo or "SXRV.DE"
    if isinstance(target, (list, tuple)):
        target = target[0]

    # 1. Trading 212 live price (EUR, real-time if market open + position exists)
    t212_price = get_t212_price(target)
    if t212_price:
        logger.info(f"Using T212 live price for {target}: {t212_price:.2f} EUR")
        return t212_price

    # 2. MarketDataManager (yfinance download)
    if MarketDataManager:
        try:
            dm = MarketDataManager(target)
            df = dm.get_price_data(force_refresh=True)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"MarketDataManager price error ({target}): {e}")

    # 3. yfinance history fallback
    try:
        import yfinance as yf

        ticker = yf.Ticker(target)
        hist = ticker.history(period="5d", timeout=10)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            logger.info(f"Using yfinance fallback price for {target}: {price:.2f} EUR")
            return price
    except Exception as e:
        logger.error(f"All price sources failed for {target}: {e}")

    raise ValueError(f"Could not retrieve price for {target} from any source")


def execute_t212_trade(
    signal,
    confidence,
    ticker=DEFAULT_TICKER,
    analysis_date=None,
    signal_source="IA_HYBRID",
):
    # Mapping du ticker Yahoo vers le ticker T212 via helper
    t212_ticker = get_t212_ticker(ticker)

    # Date pour la BDD (maintenant ou date d'analyse fournie)
    db_date = (
        analysis_date
        if analysis_date
        else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

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
            if resp.status_code == 429 or (
                resp.status_code == 400 and "TooManyRequests" in resp.text
            ):
                wait = (attempt + 1) * 2
                print(f"⚠️ Rate limit atteint, attente de {wait}s...")
                time.sleep(wait)
                continue
            return resp
        return resp

    def get_portfolio_info():
        """Vérifie le cash et les positions réelles sur Trading 212."""
        summary = safe_request(
            "GET", f"{base_url}/equity/account/summary", headers=headers
        )
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
    print("📊 VÉRIFICATION PORTEFEUILLE RÉEL :")
    print(f"   - Cash total disponible : {portfolio['cash']:.2f} €")

    # Trouver la position spécifique si elle existe
    current_pos = next(
        (p for p in portfolio["positions"] if p["instrument"]["ticker"] == t212_ticker),
        None,
    )

    if current_pos:
        print(
            f"   - Position détectée : {current_pos['quantity']} actions de {t212_ticker}"
        )
    else:
        print(f"   - Aucune position ouverte sur {t212_ticker}")

    if signal == "BUY":
        # BLOCAGE CRITIQUE : Si une position existe sur T212 OU dans notre suivi
        if current_pos or state.get("active_position"):
            if current_pos:
                print(
                    f"⚠️ Position RÉELLE déjà active pour {t212_ticker} ({current_pos['quantity']} actions). Achat ignoré."
                )
                # Resynchronisation du suivi si nécessaire
                if not state.get("active_position"):
                    print(
                        "🔄 Synchronisation du suivi local avec la position réelle..."
                    )
                    entry_price = (
                        current_pos.get("averagePrice")
                        or current_pos.get("avgPrice")
                        or (
                            current_pos["walletImpact"]["currentValue"]
                            / current_pos["quantity"]
                            if current_pos["quantity"] > 0
                            else 0.0
                        )
                    )
                    state["active_position"] = {
                        "ticker": t212_ticker,
                        "quantity": current_pos["quantity"],
                        "buy_budget": current_pos["walletImpact"]["currentValue"],
                        "entry_price_etf": entry_price,
                        "entry_price_index": entry_price,
                        "entry_time": datetime.datetime.now().isoformat(),
                    }
                    save_portfolio_state(state, t212_ticker)
            else:
                print(
                    f"⚠️ Position déjà active pour {t212_ticker} dans le suivi. Achat ignoré."
                )
            return

        # 1. Obtenir le prix le plus précis possible
        try:
            current_price = get_real_price_eur(ticker)
            # --- AJOUT : Obtenir aussi le prix de l'INDICE de référence ---
            index_ticker = (
                "^NDX"
                if "SXRV" in t212_ticker.upper()
                else "CL=F"
                if "CRUD" in t212_ticker.upper()
                else ticker
            )
            try:
                index_price = get_real_price_eur(index_ticker)
            except (ValueError, requests.RequestException, RuntimeError) as e:
                print(
                    f"⚠️ Impossible de récupérer le prix de l'indice {index_ticker}, utilisation du prix de l'ETF : {e}"
                )
                index_price = current_price
        except ValueError as e:
            print(f"❌ Impossible d'obtenir le prix : {e}")
            return
        print(
            f"🔍 CALCUL DU PRIX DU MARCHÉ : {current_price} € / action (Indice {index_ticker}: {index_price:.2f})"
        )

        # 2. Calculer la quantité
        available_cash = state.get("current_capital", 1000.0)
        if portfolio["cash"] < available_cash:
            print(
                f"⚠️ Pas assez de cash réel ({portfolio['cash']:.2f}€) pour le budget cible ({available_cash:.2f}€)."
            )

        target_budget = min(available_cash, portfolio["cash"]) * 0.99
        # Déterminer la précision selon le ticker
        precision = 2 if "CRUD" in t212_ticker.upper() else 4
        quantity = round(target_budget / current_price, precision)

        estimated_cost = quantity * current_price
        print("📊 CALCUL QUANTITÉ FRACTIONNÉE :")
        print(f"   - Budget cible : {available_cash:.2f} €")
        print(f"   - Quantité calculée : {quantity} actions (Precision: {precision})")
        print(f"   - Coût estimé : {estimated_cost:.2f} €")

        if quantity <= 0:
            print("❌ Quantité nulle ou négative, abandon.")
            return

        # 3. Passage de l'ordre
        print(f"🚀 Envoi de l'ordre d'achat de {quantity} {t212_ticker}...")
        order_data = {"ticker": t212_ticker, "quantity": quantity}
        resp = safe_request(
            "POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data
        )

        if resp.status_code in [200, 201, 202]:
            print(f"✅ Ordre placé ! Quantité : {quantity}")
            state["active_position"] = {
                "ticker": t212_ticker,
                "quantity": quantity,
                "buy_budget": estimated_cost,
                "entry_price_etf": current_price,
                "entry_price_index": index_price,
                "entry_time": datetime.datetime.now().isoformat(),
            }
            save_portfolio_state(state, t212_ticker)

            # --- Enregistrement SQLITE après confirmation ---
            if insert_transaction:
                insert_transaction(
                    date=db_date,
                    ticker=ticker,
                    type="BUY",
                    quantity=quantity,
                    price=current_price,
                    cost=estimated_cost,
                    signal_source=signal_source,
                    reason=f"T212 Order Confirmed (Index: {index_price:.2f})",
                )
        else:
            print(f"❌ Échec de l'achat : {resp.text}")

    elif signal == "SELL":
        if not state.get("active_position") and not current_pos:
            print(f"⚠️ Pas de position active pour {t212_ticker}.")
            return

        if not current_pos:
            print(
                "⚠️ Position présente dans le suivi mais INTROUVABLE sur T212. Reset du suivi."
            )
            state["active_position"] = None
            save_portfolio_state(state, t212_ticker)
            return

        # Vente de TOUTE la quantité possédée sur T212
        total_qty = current_pos["quantityAvailableForTrading"]
        current_value_eur = current_pos["walletImpact"]["currentValue"]

        print(f"📉 Vente de TOUTE la position sur {t212_ticker} ({total_qty} actions)")

        order_data = {"ticker": t212_ticker, "quantity": -total_qty}
        sell_resp = safe_request(
            "POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data
        )

        if sell_resp.status_code in [200, 201, 202]:
            print("✅ Vente effectuée.")
            # Calcul précis du nouveau capital en incluant le cash non-investi (résiduel)
            buy_cost = (
                state["active_position"]["buy_budget"]
                if state.get("active_position")
                else current_value_eur
            )

            previous_capital = state.get("current_capital", buy_cost)
            residual_cash = max(0, previous_capital - buy_cost)

            state["current_capital"] = current_value_eur + residual_cash
            state["total_realized_pl"] += current_value_eur - buy_cost

            print(f"💰 Détail capital {t212_ticker} :")
            print(f"   - Produit vente : {current_value_eur:.2f} €")
            print(f"   - Cash résiduel récupéré : {residual_cash:.2f} €")
            print(f"   - Nouveau total : {state['current_capital']:.2f} €")

            state["active_position"] = None
            save_portfolio_state(state, t212_ticker)

            # --- Enregistrement SQLITE après confirmation ---
            if insert_transaction:
                insert_transaction(
                    date=db_date,
                    ticker=ticker,
                    type="SELL",
                    quantity=total_qty,
                    price=current_value_eur / total_qty if total_qty > 0 else 0,
                    cost=current_value_eur,
                    signal_source=signal_source,
                    reason=f"T212 Confirmed Sale (P&L: {(current_value_eur - buy_cost):+.2f}€, {((current_value_eur / buy_cost) - 1):+.2%})",
                )
        else:
            print(f"❌ Erreur lors de la vente : {sell_resp.text}")


if __name__ == "__main__":
    print("Exécuteur corrigé.")
