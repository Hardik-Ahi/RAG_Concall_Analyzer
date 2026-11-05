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
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


class FixedDate:
    @classmethod
    def today(cls):
        return Date(2026, 11, 5)


class QueryParserTests(unittest.TestCase):
    def setUp(self):
        self.swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)

    def parse(self, response, query="Analyze performance."):
        self.swarm.llm = FakeLLM(response)
        state = {"user_query": query, "target_tickers": ["INFY"]}
        with patch("core_rag.agents_graph.date", FixedDate), redirect_stdout(io.StringIO()):
            result = self.swarm.node_query_parser(state)
        return result, self.swarm.llm.prompts[0]

    def test_returns_exact_temporal_filter_and_supplies_current_date(self):
        expected = {
            "all_years": False,
            "latest": False,
            "earliest": False,
            "time_list": ["2026_06", "2026_07"],
        }

        result, prompt = self.parse(
            '{"all_years": false, "latest": false, "earliest": false, '
            '"time_list": ["2026_06", "2026_07"]}',
            "Analyze performance in June and July 2026.",
        )

        self.assertEqual(result, {"time_filter": expected})
        self.assertIn("Current year and month: 2026-11.", prompt)
        self.assertIn("Expand explicit date ranges inclusively", prompt)
        self.assertIn("preceding X-1 calendar months", prompt)
        self.assertIn("all_years to true", prompt)

    def test_accepts_boolean_modes_with_empty_time_list(self):
        for field in ("all_years", "latest", "earliest"):
            with self.subTest(field=field):
                result, _ = self.parse(
                    '{"all_years": %s, "latest": %s, "earliest": %s, "time_list": []}'
                    % tuple("true" if current == field else "false" for current in (
                        "all_years", "latest", "earliest"
                    ))
                )
                self.assertTrue(result["time_filter"][field])
                self.assertEqual(result["time_filter"]["time_list"], [])

    def test_accepts_json_code_fences(self):
        result, _ = self.parse(
            '```json\n{"all_years": true, "latest": false, '
            '"earliest": false, "time_list": []}\n```'
        )

        self.assertTrue(result["time_filter"]["all_years"])

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            self.parse("not json")

    def test_rejects_extra_or_missing_keys(self):
        for response in (
            '{"all_years": false, "latest": false, "earliest": false, '
            '"time_list": ["2026_06"], "ticker": "INFY"}',
            '{"all_years": false, "latest": false, "time_list": ["2026_06"]}',
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "exactly all_years"):
                    self.parse(response)

    def test_rejects_non_boolean_flags(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.parse(
                '{"all_years": 1, "latest": false, "earliest": false, '
                '"time_list": ["2026_06"]}'
            )

    def test_preserves_overlapping_modes_and_time_list(self):
        result, _ = self.parse(
            '{"all_years": true, "latest": true, "earliest": false, '
            '"time_list": ["2026_06", "2026_07"]}'
        )

        self.assertTrue(result["time_filter"]["all_years"])
        self.assertTrue(result["time_filter"]["latest"])
        self.assertEqual(result["time_filter"]["time_list"], ["2026_06", "2026_07"])

    def test_rejects_invalid_month_entries(self):
        for timestamp in ("0000_06", "2026_00", "2026_13", "2026_6", 202606):
            with self.subTest(timestamp=timestamp):
                response = (
                    '{"all_years": false, "latest": false, "earliest": false, '
                    '"time_list": [%s]}' % json.dumps(timestamp)
                )
                with self.assertRaisesRegex(ValueError, "Invalid time_list"):
                    self.parse(response)

    def test_rejects_empty_or_nonchronological_time_lists_when_flags_are_false(self):
        for timestamps in ("[]", '["2026_06", "2026_06"]', '["2026_07", "2026_06"]'):
            with self.subTest(timestamps=timestamps):
                response = (
                    '{"all_years": false, "latest": false, "earliest": false, '
                    '"time_list": %s}' % timestamps
                )
                with self.assertRaises(ValueError):
                    self.parse(response)

    def test_allows_time_list_when_boolean_mode_is_selected(self):
        result, _ = self.parse(
            '{"all_years": true, "latest": false, "earliest": false, '
            '"time_list": ["2026_06"]}'
        )
        self.assertEqual(result["time_filter"]["time_list"], ["2026_06"])


if __name__ == "__main__":
    unittest.main()
