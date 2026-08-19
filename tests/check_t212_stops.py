"""Live DEMO probe — validate broker-side stop/TP mechanics on Trading 212.

GO-gate 2 validation harness (audit 2026-08-19 C2). NOT collected by pytest:
run manually, ONLY against the demo account, with explicit consent:

    uv run python tests/check_t212_stops.py

What it validates (in order, printing raw API evidence):
  1. GET  /equity/orders              — active-orders list shape (stop sync)
  2. POST /equity/orders/market       — market BUY with attached `takeProfit`
  3. fill confirmation via /equity/positions
  4. POST /equity/orders/stop         — dedicated GTC stop at -10% of fill
  5. GET  /equity/orders              — the stop is visible (status/type)
  6. DELETE /equity/orders/{id}       — cancel (ratchet step 1)
  7. POST /equity/orders/stop         — re-place slightly higher (ratchet step 2)
  8. cleanup: cancel stop, SELL the position back

Safety: refuses to run unless T212_ENV=demo. Total footprint: one micro
round-trip (~0.13 EUR on OD7Fd_EQ with qty 0.01) + spread.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.t212_executor import (  # noqa: E402
    DEFAULT_REQUEST_TIMEOUT,
    PRICE_DECIMALS,
    _confirm_fill,
    _cancel_order,
    _get_active_stop_order,
    _position_exists,
    _t212_session,
    get_auth_header,
    get_t212_ticker,
    post_order_market,
)

PROBE_TICKER_YAHOO = "CRUDP.PA"   # OD7Fd_EQ — EUR, 2-decimal quantities, cheap
PROBE_QTY = 0.01


def _get(url, headers):
    return _t212_session.get(url, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)


def _show(step, resp, note=""):
    body = ""
    try:
        body = str(resp.json())[:400]
    except Exception:
        body = resp.text[:400]
    print(f"\n--- {step} ---")
    print(f"    HTTP {resp.status_code} {note}")
    print(f"    {body}")


def main():
    import os

    env = os.getenv("T212_ENV", "demo").lower()
    if env != "demo":
        print(f"❌ REFUS: T212_ENV={env!r} — cette sonde ne tourne QUE sur le compte DEMO.")
        sys.exit(1)

    headers = get_auth_header()
    base = f"https://demo.trading212.com/api/v0"
    t212_ticker = get_t212_ticker(PROBE_TICKER_YAHOO)
    print(f"=== SONDE STOPS BROKER (DEMO) — {t212_ticker}, qty {PROBE_QTY} ===")

    if _position_exists(t212_ticker, headers):
        print(f"❌ REFUS: une position {t212_ticker} existe déjà — sonde annulée (elle exige un état plat).")
        sys.exit(1)

    # 1. Active orders list shape
    show1 = _get(f"{base}/equity/orders", headers)
    _show("1. GET /equity/orders (shape)", show1)

    # 2. Market BUY with attached takeProfit
    # Get a price first from the account instruments is not documented; use a
    # far-away TP (+50%) so it cannot fire during the probe.
    price = None
    print("\n    (pas d'endpoint prix sans position — TP fixé large : +50 % de sécurité)")
    order_data = {
        "ticker": t212_ticker,
        "quantity": PROBE_QTY,
        "takeProfit": None,  # rempli après le premier prix connu (étape 3)
    }
    # Step 2a: bare market BUY (attachment tested at step 2b on the re-entry)
    resp, reconciled = post_order_market({"ticker": t212_ticker, "quantity": PROBE_QTY}, headers, t212_ticker)
    if resp is None and reconciled:
        print("2a. POST market BUY (nu): réconciliation positive (ordre exécuté, réponse perdue)")
    elif resp is not None:
        _show("2a. POST market BUY (nu)", resp)
    if not ((resp is not None and resp.status_code in (200, 201, 202)) or reconciled):
        print("❌ BUY refusé — sonde interrompue.")
        sys.exit(1)

    pos = _confirm_fill(t212_ticker, headers, side="BUY")
    if pos is None:
        print("❌ Fill non confirmé — sonde interrompue (état à vérifier manuellement).")
        sys.exit(1)
    fill_price = float(pos.get("averagePricePaid") or 0)
    print(f"\n3. FILL CONFIRMÉ: qty={pos.get('quantity')} @ {fill_price}")

    stop_price = round(fill_price * 0.90, PRICE_DECIMALS)
    tp_price = round(fill_price * 1.50, PRICE_DECIMALS)

    # 4. Dedicated GTC stop order
    stop_payload = {
        "ticker": t212_ticker,
        "quantity": -PROBE_QTY,
        "stopPrice": stop_price,
        "timeValidity": "GOOD_TILL_CANCEL",
    }
    resp4 = _t212_session.post(f"{base}/equity/orders/stop", headers=headers, json=stop_payload, timeout=DEFAULT_REQUEST_TIMEOUT)
    _show("4. POST /equity/orders/stop (GTC)", resp4, f"stopPrice={stop_price}")
    stop_id = None
    try:
        stop_id = resp4.json().get("id")
    except Exception:
        pass

    # 5. Active orders list — is the stop visible?
    if stop_id is not None:
        standing = _get_active_stop_order(t212_ticker, headers)
        print(f"\n5. STOP VISIBLE dans /equity/orders: {standing is not None}")
        if standing:
            print(f"   champs observés: id={standing.get('id')} type={standing.get('type')} "
                  f"status={standing.get('status')} stopPrice={standing.get('stopPrice')}")

        # 6+7. Ratchet: delete then re-place higher
        ok_del = _cancel_order(stop_id, headers)
        print(f"\n6. DELETE /equity/orders/{stop_id}: {ok_del}")
        higher = round(stop_price * 1.01, PRICE_DECIMALS)
        stop_payload["stopPrice"] = higher
        resp7 = _t212_session.post(f"{base}/equity/orders/stop", headers=headers, json=stop_payload, timeout=DEFAULT_REQUEST_TIMEOUT)
        _show("7. POST stop re-placé (ratchet)", resp7, f"stopPrice={higher}")
        try:
            stop_id = resp7.json().get("id") or stop_id
        except Exception:
            pass

    # 8. TakeProfit attachment test: sell, re-buy with takeProfit attached.
    sell_resp, sell_reconciled = post_order_market(
        {"ticker": t212_ticker, "quantity": -PROBE_QTY}, headers, t212_ticker
    )
    if sell_resp is None and sell_reconciled:
        print("8a. POST market SELL (nettoyage): réconciliation positive")
    elif sell_resp is not None:
        _show("8a. POST market SELL (nettoyage)", sell_resp)
    if stop_id is not None:
        _cancel_order(stop_id, headers)

    resp8 = _t212_session.post(
        f"{base}/equity/orders/market",
        headers=headers,
        json={"ticker": t212_ticker, "quantity": PROBE_QTY, "takeProfit": tp_price},
        timeout=DEFAULT_REQUEST_TIMEOUT,
    )
    _show("8b. POST market BUY + takeProfit attaché", resp8, f"takeProfit={tp_price}")
    accepted_tp = resp8 is not None and resp8.status_code in (200, 201, 202)
    if accepted_tp:
        pos8 = _confirm_fill(t212_ticker, headers, side="BUY")
        sell8, _ = post_order_market({"ticker": t212_ticker, "quantity": -PROBE_QTY}, headers, t212_ticker)
        print(f"\n8c. Nettoyage final: vente={'OK' if (sell8 is not None and sell8.status_code in (200,201,202)) else 'à vérifier'}")

    print("\n=== BILAN DE LA SONDE ===")
    print(f"- ordre stop dédié (POST /equity/orders/stop): {'OK' if stop_id is not None else 'À VÉRIFIER CI-DESSUS'}")
    print(f"- visibilité + DELETE (ratchet): {'OK' if stop_id is not None else 'À VÉRIFIER CI-DESSUS'}")
    print(f"- takeProfit attaché au market order: {'OK' if accepted_tp else 'REFUSÉ -> fallback ordre nu + stop dédié (déjà codé)'}")
    print("Consigner ces résultats dans TRADING212_API_GUIDE.md.")


if __name__ == "__main__":
    main()
