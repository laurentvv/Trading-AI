import logging
import requests
import json
import pandas as pd

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen:latest" # Using qwen as requested, can be made configurable

def construct_llm_prompt(latest_data: pd.Series) -> str:
    """
    Construit un prompt détaillé pour le LLM à partir des dernières données de marché.
    """
    # Extraire les données clés les plus récentes
    # Using .iloc[0] because latest_data is a DataFrame with one row
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
    Répondez UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après.
    Le JSON doit avoir la structure suivante:
    {{
      "signal": "BUY|SELL|HOLD",
      "confidence": <float>,
      "analysis": "<votre analyse ici>"
    }}
    """
    return prompt.strip()

def get_llm_decision(latest_data: pd.DataFrame) -> dict:
    """
    Interroge le LLM via Ollama pour obtenir une décision de trading structurée.
    """
    logger.info("Interrogation du LLM pour une décision de trading...")

    prompt = construct_llm_prompt(latest_data)

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Ollama will ensure the output is valid JSON
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP 4xx/5xx

        # Le contenu de la réponse est une chaîne JSON, il faut la parser une fois.
        response_data = response.json()

        # La réponse de l'API Ollama encapsule le JSON généré dans la clé 'response'.
        # Il faut donc parser cette chaîne JSON une seconde fois.
        llm_output = json.loads(response_data.get('response', '{{}}'))

        # Validation de la structure de la réponse du LLM
        required_keys = ["signal", "confidence", "analysis"]
        if not all(key in llm_output for key in required_keys):
            logger.error(f"La réponse du LLM est mal formée: {llm_output}")
            raise ValueError("Réponse du LLM invalide ou mal formée.")

        logger.info("Décision du LLM reçue et validée.")
        return llm_output

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de communication avec l'API Ollama: {e}")
        # Retourner une décision neutre en cas d'échec de la communication
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "analysis": f"Erreur de communication avec l'API Ollama: {e}"
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Erreur de décodage ou de validation de la réponse JSON du LLM: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "analysis": f"La réponse du LLM n'était pas un JSON valide ou était mal formée: {e}"
        }
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'interrogation du LLM: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "analysis": f"Erreur inattendue: {e}"
        }
