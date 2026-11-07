import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import date as Date
from types import SimpleNamespace
from unittest.mock import patch

from core_rag.agents_graph import LangGraphFinancialSwarm


class FakeLLM:
    def __init__(self, content):
        self.responses = content if isinstance(content, list) else [content]
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected extra LLM invocation.")
        return SimpleNamespace(content=self.responses.pop(0))


class FixedDate:
    @classmethod
    def today(cls):
        return Date(2026, 11, 5)


class QueryParserTests(unittest.TestCase):
    def setUp(self):
        self.swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)

    def parse(self, responses, query="Analyze performance."):
        self.swarm.llm = FakeLLM(responses)
        state = {"user_query": query, "target_tickers": ["INFY"]}
        with patch("core_rag.agents_graph.date", FixedDate), redirect_stdout(io.StringIO()):
            result = self.swarm.node_query_parser(state)
        return result, self.swarm.llm.prompts

    def test_returns_exact_temporal_filter_and_supplies_current_date(self):
        expected = {
            "all_years": False,
            "latest": False,
            "earliest": False,
            "time_list": ["2026_06", "2026_07"],
        }

        result, prompts = self.parse(
            [
                '{"all_years": false, "latest": false, "earliest": false}',
                '["2026_06", "2026_07"]',
            ],
            "Analyze performance in June and July 2026.",
        )

        self.assertEqual(result, {"time_filter": expected})
        self.assertEqual(len(prompts), 2)
        self.assertIn("Current year and month: 2026-11.", prompts[0])
        self.assertIn("Current year and month: 2026-11.", prompts[1])
        self.assertIn("Analyze performance in June and July 2026.", prompts[1])
        self.assertIn("Expand explicit date ranges inclusively", prompts[1])
        self.assertIn("all 12 months", prompts[1])
        self.assertIn("preceding X-1 calendar months", prompts[1])

    def test_accepts_boolean_modes_with_empty_time_list(self):
        for field in ("all_years", "latest", "earliest"):
            with self.subTest(field=field):
                result, prompts = self.parse(
                    '{"all_years": %s, "latest": %s, "earliest": %s}'
                    % tuple("true" if current == field else "false" for current in (
                        "all_years", "latest", "earliest"
                    ))
                )
                self.assertTrue(result["time_filter"][field])
                self.assertEqual(result["time_filter"]["time_list"], [])
                self.assertEqual(len(prompts), 1)

    def test_accepts_json_code_fences(self):
        result, prompts = self.parse(
            '```json\n{"all_years": true, "latest": false, "earliest": false}\n```'
        )

        self.assertTrue(result["time_filter"]["all_years"])
        self.assertEqual(len(prompts), 1)

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            self.parse("not json")

    def test_rejects_extra_or_missing_keys(self):
        for response in (
            '{"all_years": false, "latest": false, "earliest": false, "ticker": "INFY"}',
            '{"all_years": false, "latest": false}',
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "exactly all_years, latest, and earliest"):
                    self.parse(response)

    def test_rejects_non_boolean_flags(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.parse(
                '{"all_years": 1, "latest": false, "earliest": false}'
            )

    def test_preserves_overlapping_modes_and_time_list(self):
        result, prompts = self.parse(
            '{"all_years": true, "latest": true, "earliest": false}'
        )

        self.assertTrue(result["time_filter"]["all_years"])
        self.assertTrue(result["time_filter"]["latest"])
        self.assertEqual(result["time_filter"]["time_list"], [])
        self.assertEqual(len(prompts), 1)

    def test_rejects_invalid_month_entries(self):
        for timestamp in ("0000_06", "2026_00", "2026_13", "2026_6", 202606):
            with self.subTest(timestamp=timestamp):
                response = '{"all_years": false, "latest": false, "earliest": false}'
                with self.assertRaisesRegex(ValueError, "Invalid time_list"):
                    self.parse([response, json.dumps([timestamp])])

    def test_rejects_empty_or_nonchronological_time_lists_when_flags_are_false(self):
        for timestamps in ("[]", '["2026_06", "2026_06"]'):
            with self.subTest(timestamps=timestamps):
                response = '{"all_years": false, "latest": false, "earliest": false}'
                with self.assertRaises(ValueError):
                    self.parse([response, timestamps])

    def test_rejects_non_array_second_response(self):
        response = '{"all_years": false, "latest": false, "earliest": false}'
        with self.assertRaisesRegex(ValueError, "must be a list"):
            self.parse([response, '{"time_list": ["2026_06"]}'])

    def test_rejects_malformed_second_response(self):
        response = '{"all_years": false, "latest": false, "earliest": false}'
        with self.assertRaises(ValueError):
            self.parse([response, "not json"])

    def test_second_prompt_expands_relative_period_from_current_month(self):
        response = '{"all_years": false, "latest": false, "earliest": false}'
        _, prompts = self.parse([response, '["2026_09", "2026_10", "2026_11"]'])

        self.assertIn("last X months", prompts[1])
        self.assertIn("preceding (12*X)-1 calendar months", prompts[1])
        self.assertIn("chronological order without duplicates", prompts[1])

    def test_second_prompt_is_not_invoked_when_a_mode_is_selected(self):
        response = '{"all_years": false, "latest": true, "earliest": false}'
        result, prompts = self.parse(response)

        self.assertEqual(len(prompts), 1)
        self.assertTrue(result["time_filter"]["latest"])
        self.assertEqual(result["time_filter"]["time_list"], [])


if __name__ == "__main__":
    unittest.main()
