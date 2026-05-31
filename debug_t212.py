"""
Debug script for Trading 212 BUY/SELL cycle validation.

Usage:
    python debug_t212.py status              # Show portfolio + state
    python debug_t212.py buy CRUDP.PA 500    # Buy ~500€ worth
    python debug_t212.py sell CRUDP.PA       # Sell entire position
    python debug_t212.py buy SXRV.DE 300     # Buy ~300€ worth
    python debug_t212.py reset CRUDP.PA      # Reset local state for ticker
"""
import sys
import os
import json
import base64
import logging
import datetime
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(".env.t212")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = Path("t212_portfolio_state.json")

TICKER_MAPPING = {
    "SXRV.DE": "SXRVd_EQ",
    "SXRV.FRK": "SXRVd_EQ",
    "CRUDP.PA": "CRUDl_EQ",
    "CRUDP": "CRUDl_EQ",
}

INITIAL_BUDGETS = {"SXRVd_EQ": 1000.0, "CRUDl_EQ": 1000.0}


def get_t212_key(ticker):
    return TICKER_MAPPING.get(ticker, ticker.split(".")[0])


def get_auth():
    key = os.getenv("T212_API_KEY")
    secret = os.getenv("T212_API_SECRET")
    if not key or not secret:
        print("ERROR: T212_API_KEY / T212_API_SECRET not set in .env.t212")
        sys.exit(1)
    b64 = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return {"Authorization": f"Basic {b64}"}


def get_base_url():
    env = os.getenv("T212_ENV", "demo").lower()
    return f"https://{env}.trading212.com/api/v0", env


def safe_req(method, url, headers, **kwargs):
    for attempt in range(3):
        resp = __import__("requests").request(method, url, headers=headers, **kwargs)
        if resp.status_code == 429:
            time.sleep((attempt + 1) * 2)
            continue
        return resp
    return resp


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"tickers": {}}


def save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=4)
    os.replace(str(tmp), str(STATE_FILE))


def get_ticker_state(state, t212_key):
    budget = INITIAL_BUDGETS.get(t212_key, 1000.0)
    if t212_key not in state["tickers"]:
        state["tickers"][t212_key] = {
            "initial_budget": budget,
            "current_capital": budget,
            "total_realized_pl": 0.0,
            "active_position": None,
        }
        save_state(state)
    return state["tickers"][t212_key]


def get_price(ticker_yahoo):
    import yfinance as yf
    t = yf.Ticker(ticker_yahoo)
    hist = t.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price for {ticker_yahoo}")
    return float(hist["Close"].iloc[-1])


def cmd_status(args):
    base_url, env = get_base_url()
    headers = get_auth()
    print(f"\n{'='*60}")
    print(f"  Trading 212 STATUS ({env.upper()})")
    print(f"{'='*60}")

    summary = safe_req("GET", f"{base_url}/equity/account/summary", headers=headers)
    if summary.status_code == 200:
        data = summary.json()
        cash = data.get("cash", {})
        print(f"  Cash available : {cash.get('availableToTrade', 0):.2f} €")
        print(f"  Cash total     : {cash.get('total', 0):.2f} €")
    else:
        print(f"  ERROR: {summary.status_code} {summary.text[:200]}")

    positions = safe_req("GET", f"{base_url}/equity/positions", headers=headers)
    if positions.status_code == 200:
        for pos in positions.json():
            ticker = pos["instrument"]["ticker"]
            qty = pos["quantity"]
            val = pos["walletImpact"]["currentValue"]
            avg = pos.get("averagePrice", pos.get("avgPrice", 0))
            pnl = val - (avg * qty)
            print(f"  Position: {ticker} | {qty} shares | {val:.2f}€ | P&L: {pnl:+.2f}€")
        if not positions.json():
            print("  No open positions")
    else:
        print(f"  Positions ERROR: {positions.status_code}")

    state = load_state()
    print("\n  Local state:")
    for k, v in state.get("tickers", {}).items():
        pos = v.get("active_position")
        print(f"    {k}: capital={v.get('current_capital', 0):.2f}€ | "
              f"budget={v.get('initial_budget', 0):.2f}€ | "
              f"P&L={v.get('total_realized_pl', 0):+.2f}€ | "
              f"pos={'YES' if pos else 'NO'}")


def cmd_buy(args):
    if len(args) < 2:
        print("Usage: debug_t212.py buy <TICKER> <AMOUNT_EUR>")
        sys.exit(1)

    ticker_yahoo = args[0]
    amount_eur = float(args[1])
    t212_key = get_t212_key(ticker_yahoo)

    base_url, env = get_base_url()
    headers = get_auth()
    print(f"\n{'='*60}")
    print(f"  BUY {t212_key} for ~{amount_eur:.2f}€ ({env.upper()})")
    print(f"{'='*60}")

    # Check no existing position
    positions = safe_req("GET", f"{base_url}/equity/positions", headers=headers)
    if positions.status_code == 200:
        for pos in positions.json():
            if pos["instrument"]["ticker"] == t212_key:
                print(f"  ERROR: Position already exists ({pos['quantity']} shares)")
                return
    else:
        print(f"  WARNING: Could not verify positions ({positions.status_code})")

    # Check cash
    summary = safe_req("GET", f"{base_url}/equity/account/summary", headers=headers)
    if summary.status_code == 200:
        cash = summary.json().get("cash", {}).get("availableToTrade", 0)
        print(f"  Cash available: {cash:.2f}€")
        if cash < amount_eur * 0.96:
            print("  ERROR: Not enough cash")
            return

    # Get price
    price = get_price(ticker_yahoo)
    print(f"  Price: {price:.4f}€ (yfinance)")

    precision = 2 if "CRUD" in t212_key.upper() else 4
    budget = min(amount_eur, cash) * 0.95
    quantity = round(budget / price, precision)
    cost = quantity * price

    print(f"  Quantity: {quantity} (precision: {precision})")
    print(f"  Est. cost: {cost:.2f}€")

    if quantity <= 0:
        print("  ERROR: Quantity is 0")
        return

    confirm = input("\n  Confirm BUY? (y/N): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order_data = {"ticker": t212_key, "quantity": quantity}
    resp = safe_req("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)

    print(f"\n  Response: {resp.status_code}")
    if resp.status_code in [200, 201, 202]:
        print(f"  SUCCESS: Bought {quantity} {t212_key}")
        print(f"  Response: {resp.json()}")

        state = load_state()
        ts = get_ticker_state(state, t212_key)
        ts["active_position"] = {
            "ticker": t212_key,
            "quantity": quantity,
            "buy_budget": cost,
            "entry_price_etf": price,
            "entry_price_index": price,
            "entry_time": datetime.datetime.now().isoformat(),
        }
        ts["current_capital"] = cost
        save_state(state)
        print("  State saved.")
    else:
        print(f"  FAILED: {resp.text}")


def cmd_sell(args):
    if not args:
        print("Usage: debug_t212.py sell <TICKER>")
        sys.exit(1)

    ticker_yahoo = args[0]
    t212_key = get_t212_key(ticker_yahoo)

    base_url, env = get_base_url()
    headers = get_auth()
    print(f"\n{'='*60}")
    print(f"  SELL {t212_key} ({env.upper()})")
    print(f"{'='*60}")

    # Find position
    positions = safe_req("GET", f"{base_url}/equity/positions", headers=headers)
    if positions.status_code != 200:
        print(f"  ERROR: Could not fetch positions ({positions.status_code})")
        return

    pos = None
    for p in positions.json():
        if p["instrument"]["ticker"] == t212_key:
            pos = p
            break

    if not pos:
        print(f"  ERROR: No position found for {t212_key}")
        return

    qty = pos["quantityAvailableForTrading"]
    val = pos["walletImpact"]["currentValue"]
    avg = float(pos.get("averagePrice", pos.get("avgPrice", 0)))
    pnl = val - (avg * qty)

    print(f"  Quantity: {qty} shares")
    print(f"  Value: {val:.2f}€")
    print(f"  Avg price: {avg:.4f}€")
    print(f"  P&L: {pnl:+.2f}€ ({(pnl / (avg * qty) * 100) if avg * qty > 0 else 0:+.2f}%)")

    confirm = input("\n  Confirm SELL all? (y/N): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    order_data = {"ticker": t212_key, "quantity": -qty}
    resp = safe_req("POST", f"{base_url}/equity/orders/market", headers=headers, json=order_data)

    print(f"\n  Response: {resp.status_code}")
    if resp.status_code in [200, 201, 202]:
        print(f"  SUCCESS: Sold {qty} {t212_key}")
        print(f"  Response: {resp.json()}")

        state = load_state()
        ts = state["tickers"].get(t212_key, {})
        prev_capital = ts.get("current_capital", val)
        buy_cost = ts.get("active_position", {}).get("buy_budget", val)
        residual = max(0, prev_capital - buy_cost)

        ts["current_capital"] = val + residual
        ts["active_position"] = None
        ts["total_realized_pl"] = ts.get("total_realized_pl", 0) + (val - buy_cost)
        state["tickers"][t212_key] = ts
        save_state(state)
        print(f"  State saved. New capital: {ts['current_capital']:.2f}€")
    else:
        print(f"  FAILED: {resp.text}")


def cmd_reset(args):
    if not args:
        print("Usage: debug_t212.py reset <TICKER>")
        sys.exit(1)
    t212_key = get_t212_key(args[0])
    state = load_state()
    budget = INITIAL_BUDGETS.get(t212_key, 1000.0)
    state["tickers"][t212_key] = {
        "initial_budget": budget,
        "current_capital": budget,
        "total_realized_pl": 0.0,
        "active_position": None,
    }
    save_state(state)
    print(f"Reset {t212_key} -> {budget:.0f}€")


COMMANDS = {
    "status": cmd_status,
    "buy": cmd_buy,
    "sell": cmd_sell,
    "reset": cmd_reset,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
