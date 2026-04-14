import unittest
import pandas as pd
import json
import sys
from pathlib import Path

# Ajouter le chemin racine pour importer src
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_client import construct_llm_prompt, get_llm_decision, get_visual_llm_decision
from unittest.mock import patch, MagicMock

class TestLLMPrompts(unittest.TestCase):
    def setUp(self):
        # Données de test réalistes
        self.sample_data = pd.DataFrame([{
            'Close': 15230.50,
            'RSI': 42.15,
            'MACD': 12.5,
            'MACD_Signal': 15.2,
            'BB_Position': 0.35,
            'Trend_Short': 1,
            'Trend_Long': -1
        }])
        self.headlines = ["US Inflation data exceeds expectations", "Tech stocks rally on AI optimism"]
        self.web_context = "OPEC+ signal possible supply cuts in June meeting."
        self.vg_indicators = {
            'HL_OIL_funding': -0.015,
            'HL_OIL_oi': 1250000.0
        }

    def test_construct_llm_prompt_content(self):
        """Affiche et vérifie le prompt texte généré."""
        prompt = construct_llm_prompt(
            self.sample_data, 
            self.headlines, 
            self.web_context, 
            self.vg_indicators
        )
        
        print("\n" + "="*50)
        print("🔍 ANALYSE DU PROMPT TEXTUEL GÉNÉRÉ :")
        print("="*50)
        print(prompt)
        print("="*50)
        
        # Vérifications
        self.assertIn("Close Price: 15230.50", prompt)
        self.assertIn("RSI (14): 42.15", prompt)
        self.assertIn("Speculative Sentiment (Hyperliquid OIL Perps):", prompt)

    @patch('src.llm_client.requests.post')
    def test_visual_llm_prompt(self, mock_post):
        """Affiche et vérifie le prompt visuel."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': json.dumps({
                "signal": "HOLD", 
                "confidence": 0.65, 
                "analysis": "Visual double bottom detected on H1 chart."
            })
        }
        mock_post.return_value = mock_response
        
        # Création d'une image factice
        dummy_path = Path("dummy_chart.png")
        dummy_path.write_bytes(b"fake_binary_data_for_test")
            
        get_visual_llm_decision(dummy_path)
        
        # Récupération du payload envoyé à Ollama
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        
        print("\n" + "="*50)
        print("🖼️ ANALYSE DU PROMPT VISUEL GÉNÉRÉ :")
        print("="*50)
        print(payload['prompt'])
        print("\n[Réponse brute attendue du modèle] :")
        print(mock_response.json.return_value['response'])
        print("="*50)
        
        self.assertIn("geometric patterns", payload['prompt'])
        if dummy_path.exists(): dummy_path.unlink()

    @patch('src.llm_client.requests.post')
    def test_llm_search_query_logic(self, mock_post):
        """Note: La génération de query de recherche est dans web_researcher.py, 
        mais nous vérifions ici comment le LLM traite le contexte web."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': json.dumps({"signal": "BUY", "confidence": 0.88, "analysis": "Web context confirms bullish bias."})}
        mock_post.return_value = mock_response
        
        with patch('src.llm_client.time.sleep'):
            result = get_llm_decision(self.sample_data, web_context=self.web_context)
            
        print("\n" + "="*50)
        print("🌐 RÉSULTAT DU TRAITEMENT DU CONTEXTE WEB :")
        print("="*50)
        print(f"Signal: {result['signal']} | Confidence: {result['confidence']}")
        print(f"Analysis: {result['analysis']}")
        print("="*50)
        
        self.assertEqual(result['signal'], "BUY")

if __name__ == '__main__':
    unittest.main()
