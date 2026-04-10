import time
import json
import os
from src.t212_executor import execute_t212_trade, load_portfolio_state, get_auth_header
import requests
from dotenv import load_dotenv

load_dotenv(".env.t212")

def print_status(message):
    print(f"\n[TEST] {message}")

def show_t212_summary():
    env = os.getenv("T212_ENV", "demo").lower()
    base_url = f"https://{env}.trading212.com/api/v0"
    headers = get_auth_header()
    
    # Résumé compte
    summary_resp = requests.get(f"{base_url}/equity/account/summary", headers=headers)
    if summary_resp.status_code == 200:
        summary = summary_resp.json()
        # On vérifie la structure (parfois c'est à la racine, parfois dans un objet)
        cash = summary.get("cash", {}).get("availableToTrade", "N/A")
        total = summary.get("totalValue", "N/A")
        print(f"💰 Cash disponible : {cash} € | Valeur Totale : {total} €")
    
    # Positions réelles
    pos_resp = requests.get(f"{base_url}/equity/positions", headers=headers)
    if pos_resp.status_code == 200:
        positions = pos_resp.json()
        if positions:
            for p in positions:
                instr = p.get('instrument', {})
                print(f"📈 Position : {instr.get('name')} | {p.get('quantity')} actions | Valeur : {p.get('walletImpact', {}).get('currentValue')} €")
        else:
            print("Empty portfolio (Aucune position réelle).")

def run_test():
    # 0. Suppression de l'historique et réinitialisation
    print_status("NETTOYAGE DE L'HISTORIQUE...")
    db_files = ["trading_history.db", "model_performance.db", "performance_monitor.db"]
    for db in db_files:
        if os.path.exists(db):
            os.remove(db)
            print(f"🗑️ Base de données {db} supprimée.")
    
    if os.path.exists("t212_portfolio_state.json"):
        os.remove("t212_portfolio_state.json")
        print("🗑️ Fichier d'état t212_portfolio_state.json supprimé.")

    state = {
        "initial_budget": 1000.0,
        "current_capital": 1000.0,
        "total_realized_pl": 0.0,
        "active_position": None
    }
    with open("t212_portfolio_state.json", 'w') as f:
        json.dump(state, f, indent=4)
    print("✨ Nouvel état initialisé à 1000€.")
    
    print_status("ÉTAPE 1 : ACHAT DE 1000€")
    execute_t212_trade("BUY", 1.0) # Confidence 1.0 pour le test
    
    # Vérifier si l'achat a réussi dans le suivi
    state = load_portfolio_state()
    if not state.get("active_position"):
        print("❌ L'achat a échoué. Arrêt du test.")
        return
    
    print_status("VÉRIFICATION APRÈS ACHAT")
    show_t212_summary()
    
    wait_minutes = 5
    print_status(f"ÉTAPE 2 : ATTENTE DE {wait_minutes} MINUTES...")
    for i in range(wait_minutes, 0, -1):
        print(f"⏳ Temps restant : {i} minute(s)...")
        time.sleep(60)

    
    print_status("VÉRIFICATION AVANT VENTE")
    show_t212_summary()
    
    print_status("ÉTAPE 3 : VENTE DE LA TOTALITÉ")
    execute_t212_trade("SELL", 1.0)
    
    print_status("RÉSULTAT FINAL")
    show_t212_summary()
    
    final_state = load_portfolio_state()
    print(f"\n✅ Test terminé.")
    print(f"Capital final IA : {final_state['current_capital']:.2f} €")
    print(f"Profit/Perte réalisé : {final_state['total_realized_pl']:.2f} €")

if __name__ == "__main__":
    run_test()
