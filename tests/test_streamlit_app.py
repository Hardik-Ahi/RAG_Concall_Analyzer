import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from streamlit_app import (
    CHILD_METADATA_SCHEMA,
    list_pdf_names,
    load_parent_lookup_summary,
    run_analysis_query,
)


class StreamlitAppTests(unittest.TestCase):
    def test_lists_pdf_names_recursively_without_other_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "financial_data" / "INFY").mkdir(parents=True)
            (root / "financial_data" / "INFY" / "call.pdf").touch()
            (root / "financial_data" / "INFY" / "notes.txt").touch()
            (root / "financial_data" / "TCS").mkdir()
            (root / "financial_data" / "TCS" / "another.PDF").touch()

            self.assertEqual(list_pdf_names(root), ["another.PDF", "call.pdf"])

    def test_summarizes_parent_lookup_without_returning_parent_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "parent_lookup.json").write_text(
                json.dumps({"parent-1": "Private transcript text"}),
                encoding="utf-8",
            )

            summary = load_parent_lookup_summary(root)

            self.assertEqual(summary, {"exists": True, "entry_count": 1})
            self.assertNotIn("Private transcript text", str(summary))

    def test_rejects_non_string_parent_lookup_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "parent_lookup.json").write_text(
                json.dumps({"parent-1": 123}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "string parent IDs"):
                load_parent_lookup_summary(root)

    def test_schema_includes_all_indexed_child_metadata(self):
        self.assertEqual(
            set(CHILD_METADATA_SCHEMA),
            {
                "parent_id",
                "ticker",
                "year",
                "month",
                "index",
                "speaker",
                "speaker_number",
                "page_number",
            },
        )
        self.assertTrue(all(value == "str" for value in CHILD_METADATA_SCHEMA.values()))

    def test_submitted_query_uses_existing_graph_and_fixed_portfolio(self):
        swarm = Mock()
        swarm.analyze_market_landscape.return_value = "Final report"

        report = run_analysis_query("  Analyze performance.  ", lambda: swarm)

        self.assertEqual(report, "Final report")
        swarm.analyze_market_landscape.assert_called_once_with(
            "Analyze performance.", ["INFY"]
        )

    def test_rejects_blank_query_without_creating_graph(self):
        swarm_factory = Mock()

        with self.assertRaisesRegex(ValueError, "Enter a query"):
            run_analysis_query("  ", swarm_factory)

        swarm_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
