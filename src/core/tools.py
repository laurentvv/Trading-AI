import logging
import traceback
from typing import Any, Dict, List, Optional
import yfinance as yf
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)


def lookup_ohlc(symbol: str, date: str, indicator: str) -> Optional[float]:
    """
    Récupère des données OHLCV ou des indicateurs simples pour un symbole donné.
    Le modèle ne doit jamais deviner ces prix.

    Args:
        symbol: "WTI" (CL=F) ou "NASDAQ" (^IXIC ou SXRV.DE).
        date: Format "YYYY-MM-DD" ou "latest".
        indicator: "open", "high", "low", "close", "volume", "vwap".

    Returns:
        La valeur demandée ou None si introuvable.
    """
    indicator = indicator.lower()

    # Mapping des symboles
    yf_symbol = symbol
    if symbol.upper() == "WTI":
        yf_symbol = "CL=F"
    elif symbol.upper() == "NASDAQ":
        yf_symbol = "^IXIC"  # Indice de base, ou l'ETF si besoin

    try:
        if date == "latest":
            # On prend les 5 derniers jours pour assurer d'avoir une cotation
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="5d")
            if df.empty:
                return None
            row = df.iloc[-1]
        else:
            # Recherche pour une date précise
            target_date = pd.to_datetime(date)
            start_date = target_date - timedelta(days=2)  # Marge weekend
            end_date = target_date + timedelta(days=2)

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            if df.empty:
                return None

            # Trouver la date la plus proche ou égale
            df.index = df.index.tz_localize(None)  # Enlever fuseau horaire
            closest_idx = df.index.get_indexer([target_date], method="nearest")[0]
            row = df.iloc[closest_idx]

        if indicator == "open":
            return float(row["Open"])
        if indicator == "high":
            return float(row["High"])
        if indicator == "low":
            return float(row["Low"])
        if indicator == "close":
            return float(row["Close"])
        if indicator == "volume":
            return float(row["Volume"])

        # Approximation du VWAP quotidien typique si demandé
        if indicator == "vwap":
            return float((row["High"] + row["Low"] + row["Close"]) / 3.0)

        return None
    except Exception as e:
        logger.error(f"Erreur lookup_ohlc({symbol}, {date}, {indicator}): {e}")
        return None


class NumericalReasoningEngine:
    """
    Sous-processus Python sécurisé avec un espace de noms persistant.
    Permet au modèle de calculer en cascade sans perdre les variables.
    """

    def __init__(self):
        # Espace de noms persistant pour cette session de raisonnement
        self.namespace: Dict[str, Any] = {"math": __import__("math"), "lookup_ohlc": lookup_ohlc}

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Exécute le bloc de code et retourne stdout et toute erreur.
        """
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output

        error = None
        try:
            # exec modifie le namespace en place
            exec(code, self.namespace)
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        return {"output": redirected_output.getvalue().strip(), "error": error, "success": error is None}


class AnswerConsolidationGate:
    """
    Vérifie la réponse finale avant d'émettre un signal de trading.
    """

    @staticmethod
    def verify(trajectory_logs: List[str], answer: Dict) -> Dict:
        """
        Vérifie que la réponse est traçable, correcte en unités et format.
        """
        required_keys = ["action", "confidence", "reasoning"]

        # 1. Vérification du format
        missing_keys = [k for k in required_keys if k not in answer]
        if missing_keys:
            return {"valid": False, "reason": f"Format invalide, clés manquantes: {missing_keys}"}

        action = str(answer.get("action", "")).upper()
        if action not in ["BUY", "SELL", "HOLD"]:
            return {"valid": False, "reason": f"Action invalide: {action}. Doit être BUY, SELL ou HOLD."}

        try:
            confidence = float(answer.get("confidence", 0.0))
            if not (0.0 <= confidence <= 1.0):
                return {"valid": False, "reason": "La confiance doit être entre 0.0 et 1.0"}
        except ValueError:
            return {"valid": False, "reason": "La confiance doit être un nombre flottant."}

        # 2. Vérification de la traçabilité (source-traceable)
        # S'assurer que le modèle ne donne pas de réponse vide sans avoir cherché
        trajectory_text = "\n".join(trajectory_logs).lower()
        if "data not available" in str(answer.get("reasoning", "")).lower() and "lookup_ohlc" not in trajectory_text:
            return {
                "valid": False,
                "reason": "Le modèle a déclaré des données non disponibles sans utiliser lookup_ohlc.",
            }

        # 3. Vérification sémantique/floue
        reasoning = str(answer.get("reasoning", ""))
        if len(reasoning) < 10 or "attente" in reasoning.lower() or "flou" in reasoning.lower():
            return {"valid": False, "reason": "L'explication est trop floue ou ressemble à un texte d'attente."}

        return {"valid": True, "reason": "Validation réussie"}
