import os
import base64
import requests
import json
from dotenv import load_dotenv
import unittest
from unittest.mock import patch, MagicMock

# Charger les variables d'environnement
load_dotenv(".env.t212")


def get_auth_header(api_key, api_secret):
    auth_str = f"{api_key}:{api_secret}"
    auth_bytes = auth_str.encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    return f"Basic {base64_auth}"


def check_connection():
    api_key = os.getenv("T212_API_KEY")
    api_secret = os.getenv("T212_API_SECRET")
    env = os.getenv("T212_ENV", "demo").lower()

    if not api_key or not api_secret:
        print("❌ Erreur : T212_API_KEY ou T212_API_SECRET non trouvés dans l'environnement.")
        print("Assurez-vous d'avoir créé un fichier .env.t212 avec vos identifiants.")
        return

    base_url = f"https://{env}.trading212.com/api/v0"
    headers = {
        "Authorization": get_auth_header(api_key, api_secret),
        "Content-Type": "application/json",
    }

    print(f"--- Test de connexion (Environnement: {env.upper()}) ---")

    # 1. Résumé du compte
    print("\n1. Récupération du résumé du compte...")
    try:
        response = requests.get(f"{base_url}/equity/account/summary", headers=headers)
        if response.status_code == 200:
            print("✅ Succès !")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Échec (Code: {response.status_code})")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ Erreur lors de l'appel : {str(e)}")
        return

    # 2. Consultation des positions
    print("\n2. Récupération des positions...")
    try:
        response = requests.get(f"{base_url}/equity/positions", headers=headers)
        if response.status_code == 200:
            positions = response.json()
            print(f"✅ Succès ! ({len(positions)} position(s) trouvée(s))")
            if positions:
                for pos in positions:
                    instr = pos.get("instrument", {})
                    print(f"  - {instr.get('name')} ({instr.get('ticker')}): {pos.get('quantity')} actions")
            else:
                print("  Aucune position ouverte.")
        else:
            print(f"❌ Échec (Code: {response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"❌ Erreur lors de l'appel : {str(e)}")


if __name__ == "__main__":
    check_connection()


class TestSafeRequest(unittest.TestCase):
    def setUp(self):
        # We need to import the function to test
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.t212_executor import safe_request

        self.safe_request = safe_request

    @patch("src.t212_executor._t212_session.request")
    def test_safe_request_success(self, mock_request):
        # Arrange
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_request.return_value = mock_resp

        # Act
        result = self.safe_request("GET", "http://test.com")

        # Assert
        self.assertEqual(result, mock_resp)
        mock_request.assert_called_once_with("GET", "http://test.com")

    @patch("time.sleep")
    @patch("src.t212_executor._t212_session.request")
    def test_safe_request_rate_limit(self, mock_request, mock_sleep):
        # Arrange
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200

        # Fails once, then succeeds
        mock_request.side_effect = [mock_resp_429, mock_resp_200]

        # Act
        result = self.safe_request("GET", "http://test.com")

        # Assert
        self.assertEqual(result, mock_resp_200)
        self.assertEqual(mock_request.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    @patch("src.t212_executor._t212_session.request")
    def test_safe_request_exception(self, mock_request, mock_sleep):
        # Arrange
        from requests.exceptions import Timeout

        # Always raises Timeout
        mock_request.side_effect = Timeout("Connection timed out")

        # Act
        result = self.safe_request("GET", "http://test.com")

        # Assert
        self.assertIsNone(result)
        self.assertEqual(mock_request.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("time.sleep")
    @patch("src.t212_executor._t212_session.request")
    def test_safe_request_exception_then_success(self, mock_request, mock_sleep):
        # Arrange
        from requests.exceptions import ConnectionError

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200

        # Fails twice, then succeeds
        mock_request.side_effect = [
            ConnectionError("Network unreachable"),
            ConnectionError("Network unreachable"),
            mock_resp_200,
        ]

        # Act
        result = self.safe_request("GET", "http://test.com")

        # Assert
        self.assertEqual(result, mock_resp_200)
        self.assertEqual(mock_request.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
