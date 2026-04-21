import unittest
from unittest.mock import patch, MagicMock
import json
import requests
from src.llm_client import _query_ollama


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.payload = {"model": "test-model", "prompt": "test prompt"}

    @patch("src.llm_client.requests.post")
    def test_query_ollama_success(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps(
                {"signal": "BUY", "confidence": 0.8, "analysis": "Test analysis"}
            )
        }
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["analysis"], "Test analysis")
        mock_post.assert_called_once()

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_request_exception_fallback(self, mock_post, mock_sleep):
        # Mock requests.post to raise RequestException
        mock_post.side_effect = requests.exceptions.RequestException("API error")

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_json_decode_error_fallback(self, mock_post, mock_sleep):
        # Mock successful request but invalid JSON in response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Invalid JSON string"}
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.llm_client.time.sleep")
    @patch("src.llm_client.requests.post")
    def test_query_ollama_value_error_fallback(self, mock_post, mock_sleep):
        # Mock successful request but missing required keys
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps(
                {
                    "signal": "BUY"
                    # Missing confidence and analysis
                }
            )
        }
        mock_post.return_value = mock_response

        result = _query_ollama(self.payload)

        self.assertEqual(result["signal"], "HOLD")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["analysis"], "")
        self.assertEqual(mock_post.call_count, 3)
        # self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
