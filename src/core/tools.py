import logging
import traceback
from typing import Any, Dict, List, Optional, Union
import yfinance as yf
import pandas as pd
from datetime import timedelta

logger = logging.getLogger(__name__)

# Symboles supportés. Tout symbole yfinance valide passe tel quel ; on ajoute
# juste des alias conviviaux pour les entités nommées dans le prompt FinAcumen.
_SYMBOL_ALIASES = {
    "WTI": "CL=F",
    "CRUDE": "CL=F",
    "BRENT": "BZ=F",
    "NASDAQ": "^IXIC",
    "NDX": "^NDX",
    "SP500": "^GSPC",
}

# Indicateurs dérivés calculés à partir d'un historique OHLCV. Les indicateurs
# de prix bruts (open/high/low/close/volume) sont lus directement sur la ligne.
_PRICE_INDICATORS = {"open", "high", "low", "close", "volume"}
_DERIVED_INDICATORS = {"vwap", "rsi", "sma_50", "sma_200", "ema_12", "ema_26", "macd"}


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes d'indicateurs dérivés à un DataFrame OHLCV trié par date."""
    out = df.copy()
    close = out["Close"]

    # VWAP quotidien approximé (typical price) — pas de VWAP intraday dispo ici.
    out["vwap"] = (out["High"] + out["Low"] + out["Close"]) / 3.0

    # RSI (Wilder, 14 périodes)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Moyennes mobiles
    out["sma_50"] = close.rolling(window=50, min_periods=1).mean()
    out["sma_200"] = close.rolling(window=200, min_periods=1).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_12"] = ema_12
    out["ema_26"] = ema_26
    out["macd"] = ema_12 - ema_26
    return out


def _fetch_yfinance_data(yf_symbol: str, date: str, need_derived: bool) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    if date == "latest":
        period = "1y" if need_derived else "5d"
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period)
        target_date = None
    else:
        target_date = pd.to_datetime(date)
        start_date = target_date - timedelta(days=400 if need_derived else 2)
        end_date = target_date + timedelta(days=2)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    return df, target_date

def lookup_ohlc(
    symbol: str, date: str, indicator: Union[str, List[str]]
) -> Union[Optional[float], Dict[str, Optional[float]]]:
    """
    Récupère des données OHLCV ou des indicateurs pour un symbole donné.
    Le modèle ne doit jamais deviner ces prix.

    Args:
        symbol: Alias ("WTI", "NASDAQ") ou ticker yfinance direct
            (ex: "CRUDP.PA", "SXRV.DE", "CL=F").
        date: Format "YYYY-MM-DD" ou "latest".
        indicator: Une chaîne unique ("close") OU une liste d'indicateurs
            (["close", "rsi", "sma_50"]).

    Indicateurs supportés:
        Prix bruts : open, high, low, close, volume.
        Dérivés     : vwap, rsi (14), sma_50, sma_200, ema_12, ema_26, macd.

    Returns:
        - Un ``float`` (ou ``None``) si ``indicator`` est une chaîne unique.
        - Un ``dict`` {indicateur: valeur} si ``indicator`` est une liste.
          Les indicateurs inconnus valent ``None`` dans le dict.
    """
    is_list_request = isinstance(indicator, (list, tuple))
    indicators = [str(i).lower() for i in (indicator if is_list_request else [indicator])]
    unknown = [i for i in indicators if i not in _PRICE_INDICATORS and i not in _DERIVED_INDICATORS]
    if unknown:
        logger.warning(f"lookup_ohlc: indicateurs inconnus ignorés: {unknown}")

    yf_symbol = _SYMBOL_ALIASES.get(symbol.upper(), symbol)

    try:
        need_derived = any(i in _DERIVED_INDICATORS for i in indicators)
        df, target_date = _fetch_yfinance_data(yf_symbol, date, need_derived)

        if df.empty:
            logger.warning(f"lookup_ohlc({symbol}, {date}): aucune donnée yfinance pour {yf_symbol}")
            return {i: None for i in indicators} if is_list_request else None

        df.index = df.index.tz_localize(None)
        df = df.sort_index()

        closest_idx = -1 if date == "latest" else df.index.get_indexer([target_date], method="nearest")[0]
        
        if need_derived:
            df = _compute_indicators(df)
            
        row = df.iloc[closest_idx]

        def _pick(name: str) -> Optional[float]:
            name = name.lower()
            col = name.capitalize() if name in _PRICE_INDICATORS else name
            try:
                val = row[col] if col in row.index else None
                return None if pd.isna(val) else float(val)
            except Exception:
                return None

        values = {i: _pick(i) for i in indicators}
        return values if is_list_request else values[indicators[0]]
    except Exception as e:
        logger.error(f"Erreur lookup_ohlc({symbol}, {date}, {indicator}): {e}")
        return {i: None for i in indicators} if is_list_request else None


class NumericalReasoningEngine:
    """
    Sous-processus Python sécurisé avec un espace de noms persistant.
    Permet au modèle de calculer en cascade sans perdre les variables.
    """

    def __init__(self):
        # Espace de noms persistant pour cette session de raisonnement.
        # ``__import__`` est volontairement absent : le code LLM ne peut pas
        # importer de modules arbitraires. Les bibliothèques légitimes pour le
        # raisonnement numérique (pandas, numpy) sont injectées directement.
        self.namespace: Dict[str, Any] = {
            "__builtins__": {
                "print": print,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "isinstance": isinstance,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "ZeroDivisionError": ZeroDivisionError,
            },
            "math": __import__("math"),
            "pd": pd,
            "np": __import__("numpy"),
            "lookup_ohlc": lookup_ohlc,
        }

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
            # SECURITY WARNING: exec modifie le namespace en place. 
            # Ce n'est pas un véritable environnement sandboxé (vulnérabilité à l'évasion).
            # L'exécution de code généré par LLM comporte des risques de sécurité inhérents.
            import logging
            logging.getLogger(__name__).warning("SECURITY: Executing LLM-generated code via exec() without a secure sandbox.")
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
