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
            return None
        except FileNotFoundError:
            return None
    return None


# Ajouter le chemin pour importer les modules du projet
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.data import MarketDataManager
    from src.database import insert_transaction, insert_portfolio_state
    from src.adaptive_weight_manager import AdaptiveWeightManager
except ImportError:
    MarketDataManager = None
    insert_transaction = None
    insert_portfolio_state = None
    AdaptiveWeightManager = None

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
# Budget initial par ticker T212 (en EUR)
INITIAL_BUDGETS = {
    "SXRVd_EQ": 1000.0,
    "SXRV_EQ": 1000.0,
    "CRUDl_EQ": 1000.0,
}
DEFAULT_INITIAL_BUDGET = 1000.0

# --- Exit-strategy thresholds (June 2026 exit-strategy audit) ---
# Four complementary exit mechanisms, evaluated unconditionally before the
# normal BUY/SELL logic in execute_t212_trade. Order of priority:
#   1. hard stop-loss   (advanced_risk_manager, forces SELL + bypasses guard)
#   2. take-profit      (direct +8% gain target)
#   3. trailing stop    (existing, -3% from peak, secures gains)
#   4. time-stop        (15 calendar days -> force exit evaluation)
TAKE_PROFIT_TARGET = 0.08   # Realized+latent gain >= +8% -> SELL to lock profit
MAX_HOLDING_DAYS = 15       # Stale-position threshold (calendar days)
TIME_STOP_SOFT_LOSS = 0.05  # Below entry by less than this, time-stop still sells


def get_t212_ticker(ticker_yahoo: str) -> str:
    """Consistently maps a Yahoo ticker to a T212 instrument ticker."""
    if not ticker_yahoo:
        return DEFAULT_TICKER
    # Use mapping if available, otherwise use prefix
    return TICKER_MAPPING_T212.get(ticker_yahoo, ticker_yahoo.split(".")[0])


def _validate_and_recalibrate_entry_price(state: dict, yahoo_ticker: str) -> dict:
    """Defend against corrupted entry prices in the portfolio state.

    PROD incident (June 2026): CRUDP.PA's state carried entry_price_etf=15.27
    (a value that never existed in the price series; max was 14.36) and
    buy_budget=1081€ for 70.8 shares, while the real fill recorded in
    trading_history.db was 13.42€ (~950€). The corrupted cost basis then
    blocked every SELL via _check_sell_loss_guard (threshold = cost*0.998),
    so the position drifted to -17% with no exit possible.

    Root cause: sync_state_from_t212 trusts the broker's averagePricePaid with
    no cross-check. This guard reconciles the stored entry price/budget against
    the last recorded BUY in trading_history.db; on a >5% discrepancy it
    recalibrates from the DB (the trusted record of what was actually paid)
    and logs an ERROR. Returns the (possibly corrected) state.
    """
    pos = state.get("active_position")
    if not pos:
        return state

    stored_price = pos.get("entry_price_etf")
    if stored_price is None or stored_price <= 0:
        return state

    try:
        from src.database import get_latest_transaction
    except Exception:
        return state  # DB module unavailable — skip validation gracefully

    try:
        last = get_latest_transaction(yahoo_ticker)
    except Exception:
        return state

    if not last or last[1] != "BUY":
        return state

    # last = (date, type, quantity, price, cost)
    db_price = float(last[3])
    db_cost = float(last[4])
    db_qty = float(last[2])
    if db_price <= 0:
        return state

    discrepancy = abs(stored_price - db_price) / db_price
    if discrepancy > 0.05:
        logger.error(
            f"🚨 STATE CORRUPTION détectée pour {yahoo_ticker}: entry_price stocké "
            f"{stored_price:.4f} vs DB {db_price:.4f} (écart {discrepancy:.1%}). "
            f"Recalage sur trading_history.db."
        )
        pos["entry_price_etf"] = db_price
        pos["entry_price_index"] = db_price
        if db_qty > 0 and db_cost > 0:
            pos["buy_budget"] = db_cost
            # highest_value must stay >= buy_budget for trailing-stop math;
            # reset conservatively to the (corrected) buy cost.
            if pos.get("highest_value", 0) < db_cost:
                pos["highest_value"] = db_cost

    return state


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


def _get_t212_base_url():
    env = os.getenv("T212_ENV", "demo").lower()
    return f"https://{env}.trading212.com/api/v0"


def get_t212_positions():
    """Fetch all open positions from T212 with live prices."""
    try:
        headers = get_auth_header()
        resp = _t212_session.get(f"{_get_t212_base_url()}/equity/positions", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"T212 positions fetch failed: {e}")
    return []


def get_t212_account_summary():
    """Fetch account summary from T212 (cash, total value, P&L)."""
    try:
        headers = get_auth_header()
        resp = _t212_session.get(f"{_get_t212_base_url()}/equity/account/summary", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"T212 account summary fetch failed: {e}")
    return None


def get_t212_order_history(ticker=None, limit=50):
    """Fetch historical filled orders from T212."""
    try:
        headers = get_auth_header()
        params = f"?limit={limit}"
        if ticker:
            params += f"&ticker={ticker}"
        resp = _t212_session.get(f"{_get_t212_base_url()}/equity/history/orders{params}", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"T212 order history fetch failed: {e}")
    return {"items": []}


def sync_state_from_t212(t212_ticker):
    """
    Build portfolio state from T212 real data instead of local JSON.
    Returns a state dict compatible with the existing system, or None if T212 is unavailable.
    """
    budget = INITIAL_BUDGETS.get(t212_ticker, DEFAULT_INITIAL_BUDGET)
    positions = get_t212_positions()
    current_pos = next((p for p in positions if p["instrument"]["ticker"] == t212_ticker), None)

    state = {
        "initial_budget": budget,
        "current_capital": budget,
        "total_realized_pl": 0.0,
        "active_position": None,
        "t212_synced": True,
    }

    if current_pos:
        entry_price = float(current_pos.get("averagePricePaid", 0) or current_pos.get("averagePrice", 0))
        qty = float(current_pos.get("quantity", 0))
        current_value = float(current_pos.get("walletImpact", {}).get("currentValue", 0))
        current_price = float(current_pos.get("currentPrice", 0))
        buy_cost = entry_price * qty

        state["active_position"] = {
            "ticker": t212_ticker,
            "quantity": qty,
            "buy_budget": buy_cost,
            "entry_price_etf": entry_price,
            "entry_price_index": entry_price,
            "entry_time": current_pos.get("createdAt", datetime.datetime.now().isoformat()),
            "highest_value": max(current_value, buy_cost),
        }

        # Calculate capital: if position is open, capital = value of position
        # Realized P&L comes from order history
        state["current_capital"] = current_value
        unrealized_pl = current_value - buy_cost
        logger.info(
            f"T212 sync: {t212_ticker} | qty={qty} | entry={entry_price:.4f} | "
            f"current={current_price:.4f} | value={current_value:.2f} EUR | "
            f"unrealized P&L={unrealized_pl:+.2f} EUR"
        )
    else:
        # No position - capital stays at budget (cash not invested)
        # Check order history for realized P&L
        order_data = get_t212_order_history(ticker=t212_ticker, limit=20)
        total_pl = 0.0
        buys = []
        for item in order_data.get("items", []):
            order = item.get("order", {})
            fill = item.get("fill", {})
            if order.get("status") != "FILLED" or not fill:
                continue
            side = order.get("side", "")
            qty = float(fill.get("quantity", 0))
            price = float(fill.get("price", 0))
            if side == "BUY":
                buys.append({"qty": qty, "price": price})
            elif side == "SELL" and buys:
                buy = buys.pop(0)
                total_pl += qty * (price - buy["price"])
        state["total_realized_pl"] = total_pl
        state["current_capital"] = budget + total_pl
        logger.info(f"T212 sync: {t212_ticker} | no position | realized P&L={total_pl:+.2f} EUR")

    return state


def load_portfolio_state(ticker=None, sync=True):
    if ticker and sync:
        clean_ticker = get_t212_ticker(ticker)
        try:
            t212_state = sync_state_from_t212(clean_ticker)
            if t212_state:
                full_state = _read_with_retry(Path(STATE_FILE))
                if full_state is None:
                    full_state = {"tickers": {}}
                if "tickers" not in full_state:
                    full_state = {"tickers": {}}

                local_state = full_state["tickers"].get(clean_ticker, {})
                if t212_state.get("active_position") is None and local_state.get("active_position") is not None:
                    local_pos = local_state["active_position"]
                    entry_time_str = local_pos.get("entry_time", "")
                    try:
                        entry_dt = datetime.datetime.fromisoformat(entry_time_str)
                        age_seconds = (datetime.datetime.now() - entry_dt).total_seconds()
                        if age_seconds < 300:
                            t212_state["active_position"] = local_pos
                            logger.debug(f"Preserved local active_position for {clean_ticker} (age={age_seconds:.0f}s)")
                    except (ValueError, TypeError):
                        pass

                full_state["tickers"][clean_ticker] = t212_state
                _atomic_json_write(Path(STATE_FILE), full_state)
                return t212_state
        except Exception as e:
            logger.warning(f"T212 sync failed, falling back to local state: {e}")

    # Fallback to local JSON state
    if not os.path.exists(STATE_FILE):
        state = {"tickers": {}}
    else:
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
            _atomic_json_write(Path(STATE_FILE), state)

    if ticker:
        clean_ticker = get_t212_ticker(ticker)
        budget = INITIAL_BUDGETS.get(clean_ticker, DEFAULT_INITIAL_BUDGET)
        if clean_ticker not in state["tickers"]:
            state["tickers"][clean_ticker] = {
                "initial_budget": budget,
                "current_capital": budget,
                "total_realized_pl": 0.0,
                "active_position": None,
            }
        else:
            t_state = state["tickers"][clean_ticker]
            t_state.setdefault("initial_budget", budget)
            t_state.setdefault("current_capital", budget)
            t_state.setdefault("total_realized_pl", 0.0)
            t_state.setdefault("active_position", None)

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
        resp = _t212_session.get(f"{base_url}/equity/positions", headers=headers, timeout=5)
        if resp.status_code == 200:
            for pos in resp.json():
                if pos["instrument"]["ticker"] == t212_ticker:
                    price = float(pos["currentPrice"])
                    logger.info(f"T212 live price for {ticker_yahoo} ({t212_ticker}): {price:.2f} EUR")
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


_t212_session = requests.Session()

def safe_request(method: str, url: str, **kwargs) -> requests.Response | None:
    """
    Execute an HTTP request with error handling and retry logic.
    """
    for attempt in range(3):
        try:
            resp = _t212_session.request(method, url, **kwargs)
            if resp.status_code == 429 or (resp.status_code == 400 and "TooManyRequests" in resp.text):
                wait = (attempt + 1) * 2
                logger.warning(f"⚠️ Rate limit atteint, attente de {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            wait = (attempt + 1) * 2
            logger.warning(f"⚠️ Erreur réseau lors de la requête: {e}. Attente de {wait}s...")
            time.sleep(wait)
            continue
    logger.error("❌ Échec de la requête après 3 tentatives.")
    return None


def _get_portfolio_info(base_url: str, headers: dict) -> dict:
    """Vérifie le cash et les positions réelles sur Trading 212."""
    summary = safe_request("GET", f"{base_url}/equity/account/summary", headers=headers)
    positions = safe_request("GET", f"{base_url}/equity/positions", headers=headers)

    info = {"cash": 0.0, "positions": []}
    if summary is not None and summary.status_code == 200:
        info["cash"] = summary.json().get("cash", {}).get("availableToTrade", 0.0)
    if positions is not None and positions.status_code == 200:
        info["positions"] = positions.json()
    return info

def _evaluate_trailing_stop(state: dict, current_pos: dict, t212_ticker: str) -> str:
    """Évalue si le trailing stop doit être déclenché et met à jour le highest value."""
    if not state.get("active_position"):
        return None

    current_value_eur = current_pos["walletImpact"]["currentValue"]
    total_qty = current_pos["quantityAvailableForTrading"]
    avg_price = current_pos.get("averagePrice") or current_pos.get("avgPrice") or 0.0
    t212_buy_cost = float(avg_price) * total_qty
    state_buy_cost = state["active_position"].get("buy_budget", 0.0)
    reference_cost = (
        max(state_buy_cost, t212_buy_cost) if max(state_buy_cost, t212_buy_cost) > 0 else current_value_eur
    )

    # Update highest value seen
    highest_value = state["active_position"].get("highest_value", reference_cost)
    if current_value_eur > highest_value:
        state["active_position"]["highest_value"] = current_value_eur
        save_portfolio_state(state, t212_ticker)
        highest_value = current_value_eur

    # Trailing Stop evaluation
    drop_from_peak = (highest_value - current_value_eur) / highest_value if highest_value > 0 else 0
    profit_margin = (current_value_eur - reference_cost) / reference_cost if reference_cost > 0 else 0

    if drop_from_peak >= 0.03 and profit_margin > 0.005:
        logger.warning(
            f"🚨 TRAILING STOP DÉCLENCHÉ ! Baisse de {drop_from_peak:.2%} depuis le sommet. Profit sécurisé de {profit_margin:.2%}."
        )
        return "SELL"
    return None

def _evaluate_take_profit(state: dict, current_pos: dict, t212_ticker: str) -> tuple[str | None, bool]:
    """
    Direct take-profit: force a SELL once the latent gain reaches
    TAKE_PROFIT_TARGET (+8%). Unlike the trailing stop (which only secures
    gains from the peak after a pullback), this locks in a concrete objective
    so a winning position is not held indefinitely waiting for a SELL signal
    that the biased consensus may never emit.

    Returns (signal, force_stop_loss). force_stop_loss is False here because a
    take-profit sale is always in profit (the sell-loss guard passes it).
    """
    if not state.get("active_position"):
        return None, False

    reference_cost = _position_reference_cost(current_pos, state)
    if reference_cost <= 0:
        return None, False

    current_value_eur = current_pos["walletImpact"]["currentValue"]
    profit_margin = (current_value_eur - reference_cost) / reference_cost

    if profit_margin >= TAKE_PROFIT_TARGET:
        logger.warning(
            f"💰 TAKE-PROFIT DÉCLENCHÉ ! Gain latent de {profit_margin:.2%} >= "
            f"+{TAKE_PROFIT_TARGET:.0%} sur {t212_ticker}. Sécurisation du profit."
        )
        return "SELL", False
    return None, False

def _evaluate_time_stop(state: dict, t212_ticker: str) -> tuple[str | None, bool]:
    """
    Time-stop: if a position has been held longer than MAX_HOLDING_DAYS (15
    calendar days), force an exit evaluation. `entry_time` was stored in the
    state since inception but never consumed — so positions could stagnate or
    bleed indefinitely (capital locked in dead positions).

    Exit rule once aged:
      - SELL (force_stop_loss=True) if the position is flat-to-up, or down by
        less than TIME_STOP_SOFT_LOSS (-5%) — i.e. cut the stale position
        rather than keep hoping.
      - For deeper losses the hard stop-loss (Phase 1A) already forced an
        EMERGENCY SELL, so here we only handle the stale-but-not-deeply-lost
        case.
    Returns (signal, force_stop_loss).
    """
    pos = state.get("active_position")
    if not pos:
        return None, False

    entry_time_str = pos.get("entry_time")
    if not entry_time_str:
        return None, False

    try:
        entry_dt = datetime.datetime.fromisoformat(entry_time_str)
    except (ValueError, TypeError):
        return None, False

    age_days = (datetime.datetime.now(entry_dt.tzinfo) - entry_dt).days
    if age_days < MAX_HOLDING_DAYS:
        return None, False

    logger.warning(
        f"⏱ TIME-STOP: position {t212_ticker} ouverte depuis {age_days} jours "
        f"(> {MAX_HOLDING_DAYS}). Évaluation de sortie forcée."
    )
    # The deep-loss case is already handled by the hard stop-loss upstream,
    # which forces a SELL before we reach here. For a stale position that is
    # not deeply underwater, cut it: bypass the sell-loss guard so a small
    # latent loss does not keep the dead position alive forever.
    return "SELL", True

def _position_reference_cost(current_pos: dict, state: dict) -> float:
    """Shared cost basis for take-profit / trailing-stop math (max of T212 avg
    and locally-tracked buy_budget). Returns 0.0 if no usable reference."""
    avg_price = current_pos.get("averagePrice") or current_pos.get("avgPrice") or 0.0
    total_qty = current_pos.get("quantityAvailableForTrading") or current_pos.get("quantity") or 0
    t212_buy_cost = float(avg_price) * float(total_qty)
    state_buy_cost = state.get("active_position", {}).get("buy_budget", 0.0)
    reference_cost = max(state_buy_cost, t212_buy_cost)
    return reference_cost if reference_cost > 0 else 0.0

def _execute_buy_order(state, current_pos, ticker, t212_ticker, portfolio, base_url, headers, db_date, signal_source, sizing_ratio=1.0):
    # BLOCAGE CRITIQUE : Si une position existe sur T212 OU dans notre suivi
    if current_pos or state.get("active_position"):
        if current_pos:
            logger.warning(
                f"⚠️ Position RÉELLE déjà active pour {t212_ticker} ({current_pos['quantity']} actions). Achat ignoré."
            )
            # Resynchronisation du suivi si nécessaire
            if not state.get("active_position"):
                logger.info("🔄 Synchronisation du suivi local avec la position réelle...")
                entry_price = (
                    current_pos.get("averagePrice")
                    or current_pos.get("avgPrice")
                    or (
                        current_pos["walletImpact"]["currentValue"] / current_pos["quantity"]
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
            logger.warning(f"⚠️ Position déjà active pour {t212_ticker} dans le suivi. Achat ignoré.")
        return

    # 1. Obtenir le prix le plus précis possible
    try:
        current_price = get_real_price_eur(ticker)
        # --- AJOUT : Obtenir aussi le prix de l'INDICE de référence ---
        index_ticker = (
            "^NDX" if "SXRV" in t212_ticker.upper() else "CL=F" if "CRUD" in t212_ticker.upper() else ticker
        )
        try:
            index_price = get_real_price_eur(index_ticker)
        except (ValueError, requests.RequestException, RuntimeError) as e:
            logger.warning(
                f"⚠️ Impossible de récupérer le prix de l'indice {index_ticker}, utilisation du prix de l'ETF : {e}"
            )
            index_price = current_price
    except ValueError as e:
        logger.error(f"❌ Impossible d'obtenir le prix : {e}")
        return
    logger.info(
        f"🔍 CALCUL DU PRIX DU MARCHÉ : {current_price} € / action (Indice {index_ticker}: {index_price:.2f})"
    )

    # 2. Calculer la quantité
    available_cash = state.get("current_capital", DEFAULT_INITIAL_BUDGET)
    if portfolio["cash"] < available_cash:
        logger.warning(
            f"⚠️ Pas assez de cash réel ({portfolio['cash']:.2f}€) pour le budget cible ({available_cash:.2f}€)."
        )

    target_budget = min(available_cash, portfolio["cash"]) * 0.95 * sizing_ratio
    # Déterminer la précision selon le ticker
    precision = 2 if "CRUD" in t212_ticker.upper() else 4
    quantity = round(target_budget / current_price, precision)

    estimated_cost = quantity * current_price
    logger.info("📊 CALCUL QUANTITÉ FRACTIONNÉE :")
    logger.info(f"   - Budget cible : {available_cash:.2f} €")
    logger.info(f"   - Quantité calculée : {quantity} actions (Precision: {precision})")
    logger.info(f"   - Coût estimé : {estimated_cost:.2f} €")

    if quantity <= 0:
        logger.error("❌ Quantité nulle ou négative, abandon.")
        return

    # 3. Passage de l'ordre
    logger.info(f"🚀 Envoi de l'ordre d'achat de {quantity} {t212_ticker}...")
    order_data = {"ticker": t212_ticker, "quantity": quantity}
    resp = safe_request("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)

    if resp is not None and resp.status_code in [200, 201, 202]:
        logger.info(f"✅ Ordre placé ! Quantité : {quantity}")
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
        if resp is None:
            logger.error("❌ Échec de l'achat : réseau (pas de réponse de l'API)")
        else:
            logger.error(f"❌ Échec de l'achat : {resp.text}")

def _check_sell_loss_guard(current_value_eur: float, current_pos: dict, state: dict) -> float | None:
    avg_price = current_pos.get("averagePrice") or current_pos.get("avgPrice") or 0.0
    total_qty = current_pos["quantityAvailableForTrading"]
    t212_buy_cost = float(avg_price) * total_qty
    state_buy_cost = state["active_position"]["buy_budget"] if state.get("active_position") else 0.0

    reference_cost = max(state_buy_cost, t212_buy_cost)
    if reference_cost == 0.0:
        reference_cost = current_value_eur

    if current_value_eur < reference_cost * 0.998:
        logger.warning(
            f"⚠️ VENTE BLOQUÉE : Perte potentielle détectée. Valeur actuelle: {current_value_eur:.2f}€, Coût d'achat de référence: {reference_cost:.2f}€."
        )
        return None
    return reference_cost


def _record_sell_transaction(state, current_value_eur, total_qty, ticker, db_date, signal_source, buy_cost):
    previous_capital = state.get("current_capital", buy_cost)
    residual_cash = max(0, previous_capital - buy_cost)

    state["current_capital"] = current_value_eur + residual_cash
    state["total_realized_pl"] += current_value_eur - buy_cost

    logger.info("💰 Détail capital :")
    logger.info(f"   - Produit vente : {current_value_eur:.2f} €")
    logger.info(f"   - Cash résiduel récupéré : {residual_cash:.2f} €")
    logger.info(f"   - Nouveau total : {state['current_capital']:.2f} €")

    entry_time_str = state["active_position"].get("entry_time") if state.get("active_position") else None

    state["active_position"] = None

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
    return entry_time_str


def _update_feedback_loop(entry_time_str, db_date, current_value_eur, buy_cost):
    if AdaptiveWeightManager is None:
        return
    try:
        wm = AdaptiveWeightManager()
        entry_date = entry_time_str[:10] if entry_time_str else db_date[:10]
        actual_outcome = 1 if current_value_eur > buy_cost else 0
        return_1d = (current_value_eur - buy_cost) / buy_cost if buy_cost > 0 else 0.0
        updated = wm.update_outcomes_for_date(
            date=entry_date,
            actual_outcome=actual_outcome,
            return_1d=return_1d,
        )
        if updated > 0:
            logger.info(
                f"📊 Feedback loop: updated {updated} model predictions for {entry_date} (return_1d={return_1d:+.4f})"
            )
    except Exception as fb_e:
        logger.warning(f"Feedback loop failed: {fb_e}")


def _execute_sell_order(state, current_pos, ticker, t212_ticker, base_url, headers, db_date, signal_source, force_stop_loss=False):
    if not state.get("active_position") and not current_pos:
        logger.warning(f"⚠️ Pas de position active pour {t212_ticker}.")
        return

    if not current_pos:
        logger.warning("⚠️ Position présente dans le suivi mais INTROUVABLE sur T212. Reset du suivi.")
        state["active_position"] = None
        save_portfolio_state(state, t212_ticker)
        return

    total_qty = current_pos["quantityAvailableForTrading"]
    current_value_eur = current_pos["walletImpact"]["currentValue"]

    # The sell-loss guard blocks any sale that would realize a loss > 0.2%.
    # That guard must be BYPASSED for emergency exits (stop-loss / time-stop)
    # — otherwise a position in deep drawdown could never be cut, which is
    # exactly what let CRUDP.PA drift to -17% (the stop fired but the guard
    # re-blocked the sale). The bypass is intentionally scoped to
    # force_stop_loss only; normal SELL signals still respect the guard.
    if force_stop_loss:
        logger.warning(
            f"🚨 FORCE STOP-LOSS: bypassing _check_sell_loss_guard for {t212_ticker} "
            f"(emergency exit). Current value {current_value_eur:.2f}€."
        )
    elif _check_sell_loss_guard(current_value_eur, current_pos, state) is None:
        return

    logger.info(f"📉 Vente de TOUTE la position sur {t212_ticker} ({total_qty} actions)")

    order_data = {"ticker": t212_ticker, "quantity": -total_qty}
    sell_resp = safe_request("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)

    if sell_resp is not None and sell_resp.status_code in [200, 201, 202]:
        logger.info("✅ Vente effectuée.")
        buy_cost = state["active_position"]["buy_budget"] if state.get("active_position") else current_value_eur
        entry_time_str = _record_sell_transaction(state, current_value_eur, total_qty, ticker, db_date, signal_source, buy_cost)
        save_portfolio_state(state, t212_ticker)
        _update_feedback_loop(entry_time_str, db_date, current_value_eur, buy_cost)
    else:
        if sell_resp is None:
            logger.error("❌ Erreur lors de la vente : réseau (pas de réponse de l'API)")
        else:
            logger.error(f"❌ Erreur lors de la vente : {sell_resp.text}")

def execute_t212_trade(
    signal,
    confidence,
    ticker=DEFAULT_TICKER,
    analysis_date=None,
    signal_source="IA_HYBRID",
    sizing_ratio=1.0,
):
    # Mapping du ticker Yahoo vers le ticker T212 via helper
    t212_ticker = get_t212_ticker(ticker)

    # Date pour la BDD (maintenant ou date d'analyse fournie)
    db_date = analysis_date if analysis_date else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Charger l'état spécifique au ticker (on utilise le ticker T212 comme clé)
    state = load_portfolio_state(t212_ticker)

    # Defend against corrupted entry prices (see _validate_and_recalibrate_entry_price):
    # reconcile the stored cost basis against trading_history.db before any
    # exit-strategy math runs, so a stale/ghost price cannot block a SELL.
    state = _validate_and_recalibrate_entry_price(state, ticker)

    env = os.getenv("T212_ENV", "demo").lower()
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = get_auth_header()

    if signal not in ["BUY", "SELL"]:
        return

    logger.info(f"\n--- 🤖 EXÉCUTION IA TRADING 212 ({env.upper()}) POUR {t212_ticker} ---")

    # Vérification systématique avant action
    portfolio = _get_portfolio_info(base_url, headers)
    logger.info("📊 VÉRIFICATION PORTEFEUILLE RÉEL :")
    logger.info(f"   - Cash total disponible : {portfolio['cash']:.2f} €")

    # Trouver la position spécifique si elle existe
    current_pos = next(
        (p for p in portfolio["positions"] if p["instrument"]["ticker"] == t212_ticker),
        None,
    )

    if current_pos:
        logger.info(f"   - Position détectée : {current_pos['quantity']} actions de {t212_ticker}")

        # --- UNIFIED EXIT STRATEGY (June 2026) ---
        # Evaluate the four exit mechanisms in priority order BEFORE the normal
        # BUY/SELL logic. They are UNCONDITIONAL — they trigger on position
        # state alone, regardless of the incoming consensus signal. This fixes
        # the root cause of CRUDP.PA drifting to -17%: previously the stops
        # were gated behind a SELL signal the biased consensus never emitted.
        # The first mechanism to fire wins; force_stop_loss tells the executor
        # to bypass _check_sell_loss_guard for emergency cuts.
        # NOTE: the hard stop-loss (-10%) is also enforced upstream in
        # advanced_risk_manager.get_risk_adjusted_signal (redundant by design,
        # belt-and-braces: that layer sets the signal, this layer guarantees
        # the sale is not re-blocked by the guard).
        force_stop_loss = False
        exit_reason = None

        # 1. Take-profit (+8%) — lock gains directly.
        tp_signal, _ = _evaluate_take_profit(state, current_pos, t212_ticker)
        if tp_signal:
            signal, exit_reason = tp_signal, "take-profit"

        # 2. Trailing stop (-3% from peak) — secure gains on pullback.
        if signal not in ["SELL"]:
            trailing_signal = _evaluate_trailing_stop(state, current_pos, t212_ticker)
            if trailing_signal:
                signal, exit_reason = trailing_signal, "trailing-stop"

        # 3. Time-stop (15 days) — cut stale positions; bypasses the guard.
        if signal not in ["SELL"]:
            ts_signal, ts_force = _evaluate_time_stop(state, t212_ticker)
            if ts_signal:
                signal, force_stop_loss, exit_reason = ts_signal, ts_force, "time-stop"

        if exit_reason:
            logger.info(f"🎯 Sortie forcée par {exit_reason} (priorité exit-strategy).")
    else:
        logger.info(f"   - Aucune position ouverte sur {t212_ticker}")

    if signal == "BUY":
        _execute_buy_order(state, current_pos, ticker, t212_ticker, portfolio, base_url, headers, db_date, signal_source, sizing_ratio)
    elif signal == "SELL":
        _execute_sell_order(state, current_pos, ticker, t212_ticker, base_url, headers, db_date, signal_source, force_stop_loss=force_stop_loss)


if __name__ == "__main__":
    print("Exécuteur corrigé.")
