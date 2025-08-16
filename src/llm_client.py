import logging
import requests
import json
import pandas as pd
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
TEXT_LLM_MODEL = "qwen:latest"
VISUAL_LLM_MODEL = "llava:latest" # Default visual model

def construct_llm_prompt(latest_data: pd.DataFrame) -> str:
    """
    Construit un prompt détaillé pour le LLM à partir des dernières données de marché.
    """
    data = latest_data.iloc[0]
    prompt = f"""
    Vous êtes un analyste financier expert spécialisé dans les ETF du NASDAQ.
    Votre tâche est d'analyser les données de marché suivantes pour l'ETF et de fournir une décision de trading.

    **Données de marché actuelles:**
    - Prix de clôture: {data['Close']:.2f}
    - RSI (14): {data['RSI']:.2f}
    - MACD: {data['MACD']:.4f}
    - Signal MACD: {data['MACD_Signal']:.4f}
    - Histogramme MACD: {data['MACD_Histogram']:.4f}
    - Position des Bandes de Bollinger (0-1): {data['BB_Position']:.2f}
    - Tendance à court terme (MA5 vs MA20): {'Haussière' if data['Trend_Short'] == 1 else 'Baissière' if data['Trend_Short'] == -1 else 'Neutre'}
    - Tendance à long terme (MA20 vs MA50): {'Haussière' if data['Trend_Long'] == 1 else 'Baissière' if data['Trend_Long'] == -1 else 'Neutre'}

    **Instructions:**
    1.  Analysez ces indicateurs pour déterminer la dynamique actuelle du marché (momentum, tendance, surachat/survente).
    2.  Fournissez une recommandation de trading claire: "BUY", "SELL", ou "HOLD".
    3.  Rédigez une brève analyse (2-3 phrases) pour justifier votre recommandation.
    4.  Fournissez un score de confiance pour votre décision, de 0.0 (incertain) à 1.0 (très certain).

    **Format de sortie:**
    Répondez UNIQUEMENT avec un objet JSON valide.
    {{
      "signal": "BUY|SELL|HOLD",
      "confidence": <float>,
      "analysis": "<votre analyse ici>"
    }}
    """
    return prompt.strip()

def get_llm_decision(latest_data: pd.DataFrame) -> dict:
    """
    Interroge le LLM textuel via Ollama pour obtenir une décision de trading.
    """
    logger.info("Interrogation du LLM textuel pour une décision de trading...")
    prompt = construct_llm_prompt(latest_data)
    payload = {
        "model": TEXT_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    return _query_ollama(payload)

def get_visual_llm_decision(image_path: Path) -> dict:
    """
    Interroge le LLM visuel via Ollama avec une image de graphique.
    """
    logger.info(f"Interrogation du LLM visuel avec l'image {image_path}...")

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Impossible de lire ou d'encoder l'image {image_path}: {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Erreur de lecture de l'image: {e}"}

    prompt = """
    Vous êtes un expert en analyse technique et un chartiste. Analysez le graphique financier fourni.
    Identifiez les tendances clés, les figures chartistes (patterns), et la dynamique des indicateurs (MAs, RSI, MACD, Volume).

    Fournissez votre analyse dans un objet JSON valide avec la structure suivante:
    {
      "signal": "BUY|SELL|HOLD",
      "confidence": <float between 0.0 and 1.0>,
      "analysis": "<votre analyse de 2-3 phrases sur les patterns visuels que vous avez identifiés>"
    }
    """

    payload = {
        "model": VISUAL_LLM_MODEL,
        "prompt": prompt.strip(),
        "images": [image_base64],
        "stream": False,
        "format": "json"
    }
    return _query_ollama(payload)

def _query_ollama(payload: dict) -> dict:
    """
    Fonction helper pour envoyer une requête à l'API Ollama et gérer la réponse.
    """
    model_name = payload.get("model", "unknown")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120) # Timeout plus long pour les modèles visuels
        response.raise_for_status()

        response_data = response.json()
        llm_output = json.loads(response_data.get('response', '{}'))

        required_keys = ["signal", "confidence", "analysis"]
        if not all(key in llm_output for key in required_keys):
            logger.error(f"La réponse du LLM ({model_name}) est mal formée: {llm_output}")
            raise ValueError("Réponse du LLM invalide ou mal formée.")

        logger.info(f"Décision du LLM ({model_name}) reçue et validée.")
        return llm_output

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de communication avec l'API Ollama ({model_name}): {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Erreur de communication avec l'API Ollama: {e}"}
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Erreur de décodage ou validation de la réponse JSON du LLM ({model_name}): {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"La réponse du LLM n'était pas un JSON valide ou était mal formée: {e}"}
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'interrogation du LLM ({model_name}): {e}")
        return {"signal": "HOLD", "confidence": 0.0, "analysis": f"Erreur inattendue: {e}"}
