import unittest
from unittest.mock import patch, MagicMock
from src.core.tools import AnswerConsolidationGate

class TestFinAcumen(unittest.TestCase):
    def test_consolidation_gate_valid(self):
        trajectory = ["Action: execute_python({'python_code': 'lookup_ohlc(\"WTI\", \"latest\", \"close\")'})"]
        answer = {"action": "BUY", "confidence": 0.8, "reasoning": "Price is above MA and lookup_ohlc returned valid data."}
        result = AnswerConsolidationGate.verify(trajectory, answer)
        self.assertTrue(result["valid"])

    def test_consolidation_gate_invalid_action(self):
        trajectory = []
        answer = {"action": "MAYBE", "confidence": 0.5, "reasoning": "Not sure."}
        result = AnswerConsolidationGate.verify(trajectory, answer)
        self.assertFalse(result["valid"])

if __name__ == '__main__':
    unittest.main()
