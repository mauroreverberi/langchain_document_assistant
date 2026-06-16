"""Integration tests that run the full workflow end-to-end.
they only run when an
OPENAI_API_KEY is available AND RUN_LIVE_TESTS=1 is set.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
RUN_LIVE_TESTS = os.getenv("RUN_LIVE_TESTS") == "1"

from assistant import DocumentAssistant


@unittest.skipUnless(API_KEY and RUN_LIVE_TESTS, "live tests requires OPENAI_API_KEY and RUN_LIVE_TESTS=1")
class TestWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.assistant = DocumentAssistant(openai_api_key=API_KEY, model_name="gpt-4o", temperature=0.1)

    def ask(self, question, user_id="test_user"):
        """Start a fresh session, send one message and confirm it succeeded."""
        self.assistant.start_session(user_id=user_id)
        result = self.assistant.process_message(question)
        self.assertTrue(result["success"], result.get("error"))
        self.assertTrue(result.get("response"))
        return result

    def test_qa_question(self):
        result = self.ask("Who are the two parties in the service agreement contract?")
        self.assertEqual(result["intent"]["intent_type"], "qa")

    def test_calculation_uses_the_calculator(self):
        result = self.ask("Calculate the sum of all invoice totals")
        self.assertEqual(result["intent"]["intent_type"], "calculation")

        self.assertIn("calculator", result["tools_used"])

    def test_summarization(self):
        result = self.ask("Summarize all contracts")
        self.assertEqual(result["intent"]["intent_type"], "summarization")

    def test_remembers_previous_turn(self):
        self.assistant.start_session(user_id="memory_user")
        first = self.assistant.process_message("What is the total amount in invoice INV-001?")
        self.assertTrue(first["success"], first.get("error"))
        second = self.assistant.process_message("And what about invoice INV-002?")
        self.assertTrue(second["success"], second.get("error"))
        self.assertTrue(second.get("response"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
