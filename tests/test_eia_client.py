import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eia_client import EIAClient


class TestIsOilTicker(unittest.TestCase):
    def test_oil_tickers(self):
        self.assertTrue(EIAClient.is_oil_ticker("CL=F"))
        self.assertTrue(EIAClient.is_oil_ticker("CRUDP.PA"))
        self.assertTrue(EIAClient.is_oil_ticker("BZ=F"))

    def test_nasdaq_tickers(self):
        self.assertFalse(EIAClient.is_oil_ticker("^NDX"))
        self.assertFalse(EIAClient.is_oil_ticker("SXRV.DE"))
        self.assertFalse(EIAClient.is_oil_ticker("QQQ"))


class TestEIAClientRequests(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {"EIA_API_KEY": "test_key_123"})
        self.env_patcher.start()
        self.client = EIAClient()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("eia_client.requests.get")
    def test_make_request_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "total": 2,
                "data": [
                    {"period": "2026-04-10", "value": "463804"},
                    {"period": "2026-04-03", "value": "464717"},
                ],
            }
        }
        mock_get.return_value = mock_response

        result = self.client._make_request("/test", {})
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    @patch("eia_client.requests.get")
    def test_get_refinery_utilization(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2026-04-10", "value": "85.0", "duoarea": "R10"},
                    {"period": "2026-04-10", "value": "95.0", "duoarea": "R20"},
                ]
            }
        }
        mock_get.return_value = mock_response

        with patch.object(self.client, "_get_from_cache", return_value=None):
            with patch.object(self.client, "_save_to_cache"):
                df = self.client.get_refinery_utilization(weeks=1)

        self.assertFalse(df.empty)
        self.assertEqual(df["value"].iloc[0], 90.0)  # Average of 85 and 95

    @patch("eia_client.requests.get")
    def test_get_brent_spot_price(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": {
                "data": [
                    {"period": "2026-04-15", "value": "117.0"},
                    {"period": "2026-04-14", "value": "118.5"},
                ]
            }
        }
        mock_get.return_value = mock_response

        with patch.object(self.client, "_get_from_cache", return_value=None):
            with patch.object(self.client, "_save_to_cache"):
                df = self.client.get_brent_spot_price(days=2)

        self.assertFalse(df.empty)
        self.assertEqual(df["value"].iloc[-1], 117.0)


class TestFormatForLLM(unittest.TestCase):
    def test_format_complete_data(self):
        client = EIAClient()
        data = {
            "as_of": "2026-04-16T10:00:00",
            "inventories": {"current": 460000, "wow_change": -1000},
            "imports": {"latest_value": 80000, "mom_change": 500},
            "refinery": {"current": 88.5, "wow_change": 1.2},
            "brent_spot": {"current": 117.0, "wow_change": -2.5},
        }
        text = client.format_for_llm(data)
        self.assertIn("US Crude Inventories", text)
        self.assertIn("US Crude Imports", text)
        self.assertIn("US Refinery Utilization", text)
        self.assertIn("Europe Brent Spot Price (Dated Brent)", text)
        self.assertIn("$117.00/bbl", text)


if __name__ == "__main__":
    unittest.main()
