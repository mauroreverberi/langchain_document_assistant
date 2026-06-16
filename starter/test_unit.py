"""Offline unit tests for the document assistant's building blocks.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from schemas import AnswerResponse, UserIntent
from tools import create_calculator_tool, ToolLogger
from prompts import get_chat_prompt_template
from agent import create_workflow


class TestSchemas(unittest.TestCase):

    def test_answer_response_valid(self):
        answer = AnswerResponse(question="q", answer="a", sources=["INV-001"], confidence=0.9)
        self.assertEqual(answer.answer, "a")
        self.assertEqual(answer.sources, ["INV-001"])

    def test_confidence_must_be_between_0_and_1(self):
        with self.assertRaises(ValidationError):
            AnswerResponse(question="q", answer="a", confidence=1.5)
        with self.assertRaises(ValidationError):
            AnswerResponse(question="q", answer="a", confidence=-0.1)

    def test_intent_type_only_allows_known_values(self):
        for intent in ["qa", "summarization", "calculation", "unknown"]:
            UserIntent(intent_type=intent, confidence=0.5, reasoning="test")
        # any other value must be rejected
        with self.assertRaises(ValidationError):
            UserIntent(intent_type="invalid", confidence=0.5, reasoning="test")


class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = create_calculator_tool(ToolLogger(logs_dir="./logs"))

    def test_basic_math(self):
        self.assertEqual(self.calc.invoke({"expression": "2+2"}), "4")
        self.assertEqual(self.calc.invoke({"expression": "10/4"}), "2.5")
        self.assertEqual(self.calc.invoke({"expression": "(5000+12500)*1.1"}), "19250")

    def test_result_is_a_string(self):
        self.assertIsInstance(self.calc.invoke({"expression": "2+2"}), str)

    def test_divide_by_zero(self):
        result = self.calc.invoke({"expression": "5/0"})
        self.assertIn("divide by zero", result.lower())

    def test_empty_input(self):
        result = self.calc.invoke({"expression": ""})
        self.assertIn("empty", result.lower())

    def test_rejects_code_injection(self):
        result = self.calc.invoke({"expression": "__import__('os')"})
        self.assertTrue(result.lower().startswith("error"))

    def test_rejects_letters(self):
        result = self.calc.invoke({"expression": "2 + abc"})
        self.assertTrue(result.lower().startswith("error"))

    def test_handles_comma_numbers(self):
        self.assertEqual(self.calc.invoke({"expression": "1,000+500"}), "1500")


class TestPrompts(unittest.TestCase):

    def test_every_intent_returns_a_template(self):
        for intent in ["qa", "summarization", "calculation", "unknown"]:
            self.assertIsInstance(get_chat_prompt_template(intent), ChatPromptTemplate)


class TestWorkflow(unittest.TestCase):

    def test_all_nodes_are_present(self):
        workflow = create_workflow(llm=None, tools=[])
        nodes = [n for n in workflow.get_graph().nodes if not n.startswith("__")]
        for name in ["classify_intent", "qa_agent", "summarization_agent", "calculation_agent", "update_memory"]:
            self.assertIn(name, nodes)


if __name__ == "__main__":
    unittest.main()
