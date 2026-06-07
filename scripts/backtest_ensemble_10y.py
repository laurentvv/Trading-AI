import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Ensure src module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grebenkov_model import GrebenkovTrendModel
from src.hmm_model import HMMDecisionModel
from src.enhanced_decision_engine import VincentGanneModel, EnhancedDecisionEngine, ModelDecision
from src.timesfm_model import TimesFMModel

def run_ensemble_backtest():
    print("Téléchargement des données sur 10 ans...")
    # Téléchargement des données
    tickers = {
        "QQQ": "QQQ",
        "WTI": "CL=F",
        "Brent": "BZ=F",
        "Gas": "NG=F",
        "DXY": "DX-Y.NYB",
    }
    
    data = {}
    for name, symbol in tickers.items():
        print(f"Téléchargement {name} ({symbol})...")
        df = yf.download(symbol, period="10y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df.ffill().dropna()

    df_qqq = data["QQQ"]
    if df_qqq.empty:
        print("Erreur : données QQQ manquantes.")
        return

    print(f"Données QQQ prêtes : {len(df_qqq)} jours.")

    # Initialisation des modèles
    print("Initialisation des modèles...")
    hmm_model = HMMDecisionModel(lookback=252, baum_welch_iterations=5)
    grebenkov_model = GrebenkovTrendModel()
    vincent_ganne = VincentGanneModel()
    
    # TimesFM (peut être long à charger)
    try:
        timesfm_model = TimesFMModel.get_instance()
        timesfm_available = True
        print("TimesFM 2.5 initialisé avec succès.")
    except Exception as e:
        print(f"Attention : TimesFM impossible à charger ({e}). Il sera ignoré.")
        timesfm_available = False

    engine = EnhancedDecisionEngine()

    # Paramètres du backtest
    initial_capital = 10000.0
    cash = initial_capital
    position = 0.0

    bh_cash = initial_capital
    bh_position = bh_cash / float(df_qqq.iloc[0]["Close"])
    bh_cash = 0.0

    history = []
    
    start_idx = 252 # 1 an de lookback
    total_steps = len(df_qqq) - start_idx
    
    print(f"\nDébut du backtest sur {total_steps} jours (cela va être TRÈS long avec TimesFM)...")
    start_time = time.time()

    for i in range(start_idx, len(df_qqq)):
        current_date = df_qqq.index[i]
        current_price = float(df_qqq.iloc[i]["Close"])
        
        # Fenêtre glissante
        window_qqq = df_qqq.iloc[i - 252 : i + 1].copy()
        
        # Prépare les données pour les modèles
        model_data = {
            "hist_data": window_qqq,
            "ticker": "QQQ",
            "wti_data": data["WTI"].iloc[max(0, i-252):i+1] if "WTI" in data else None,
            "nasdaq_data": window_qqq
        }
        
        decisions = []
        
        # 1. HMM
        res_hmm = hmm_model.predict(model_data)
        decisions.append(ModelDecision(
            signal=res_hmm.signal, confidence=res_hmm.confidence,
            strength=engine._normalize_signal(res_hmm.signal),
            timestamp=datetime.now(), model_name="hmm_model"
        ))
        
        # 2. Grebenkov
        res_greb = grebenkov_model.predict(model_data)
        decisions.append(ModelDecision(
            signal=res_greb.signal, confidence=res_greb.confidence,
            strength=engine._normalize_signal(res_greb.signal),
            timestamp=datetime.now(), model_name="grebenkov"
        ))
        
        # 3. Vincent Ganne
        # Construction des indicateurs macro à l'instant T
        vg_indicators = {}
        for name, key in [("WTI", "WTI_price"), ("Brent", "Brent_price"), ("Gas", "NaturalGas_price"), ("DXY", "DXY_price")]:
            if name in data and current_date in data[name].index:
                vg_indicators[key] = float(data[name].loc[current_date, "Close"])
            else:
                # Fallback au dernier prix connu
                if name in data and not data[name].empty:
                    valid_data = data[name][data[name].index <= current_date]
                    if not valid_data.empty:
                        vg_indicators[key] = float(valid_data.iloc[-1]["Close"])
                        
        res_vg = vincent_ganne.evaluate(vg_indicators)
        decisions.append(ModelDecision(
            signal=res_vg["signal"], confidence=res_vg["confidence"],
            strength=engine._normalize_signal(res_vg["signal"]),
            timestamp=datetime.now(), model_name="vincent_ganne"
        ))
        
        # 4. TimesFM
        if timesfm_available:
            try:
                res_tfm = timesfm_model.predict(window_qqq, horizon=5, ticker="QQQ")
                decisions.append(ModelDecision(
                    signal=res_tfm["signal"], confidence=res_tfm["confidence"],
                    strength=engine._normalize_signal(res_tfm["signal"]),
                    timestamp=datetime.now(), model_name="timesfm"
                ))
            except Exception:
                pass

        # Calculer le régime de marché simple
        returns = window_qqq["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        market_data = {"volatility": volatility, "rsi": 50}

        # 5. Moteur de décision
        final_decision = engine.make_enhanced_decision(
            generic_model_results=decisions,
            market_data=market_data
        )
        
        signal = final_decision.risk_adjusted_signal
        
        # Execution
        if signal in ["BUY", "STRONG_BUY"] and cash > 0:
            position = cash / current_price
            cash = 0.0
        elif signal in ["SELL", "STRONG_SELL"] and position > 0:
            cash = position * current_price
            position = 0.0

        current_value = cash + (position * current_price)
        bh_current_value = bh_cash + (bh_position * current_price)
        
        history.append({
            "Date": current_date,
            "Value": current_value,
            "BH_Value": bh_current_value,
            "Signal": signal
        })
        
        if (i - start_idx) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Jour {i - start_idx}/{total_steps} complété en {elapsed:.1f}s... Valeur: {current_value:.2f}$")

    # Résultats
    df_res = pd.DataFrame(history).set_index("Date")
    final_val = df_res.iloc[-1]["Value"]
    final_bh = df_res.iloc[-1]["BH_Value"]
    
    ret_ens = (final_val - initial_capital) / initial_capital * 100
    ret_bh = (final_bh - initial_capital) / initial_capital * 100
    
    print("\n" + "=" * 50)
    print("RÉSULTATS BACKTEST ENSEMBLE (10 ANS)")
    print("=" * 50)
    print(f"Capital Initial : {initial_capital:.2f} $")
    print(f"Ensemble Final  : {final_val:.2f} $ ({ret_ens:+.2f}%)")
    print(f"Buy & Hold Final: {final_bh:.2f} $ ({ret_bh:+.2f}%)")
    print("=" * 50)
    
    # Graphique
    plt.figure(figsize=(14, 7))
    plt.plot(df_res.index, df_res["Value"], label="Ensemble Model", color="blue", linewidth=2)
    plt.plot(df_res.index, df_res["BH_Value"], label="Buy & Hold (Baseline)", color="orange", alpha=0.7)
    plt.title("Benchmark sur 10 ans : Multi-Models Ensemble vs Buy & Hold (QQQ)")
    plt.xlabel("Date")
    plt.ylabel("Valeur du Portefeuille ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = Path("ensemble_benchmark_10y.png")
    plt.savefig(out_path)
    print(f"\nGraphique généré : {out_path.absolute()}")

if __name__ == "__main__":
    run_ensemble_backtest()
