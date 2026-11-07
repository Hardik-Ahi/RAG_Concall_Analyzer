import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from streamlit_app import (
    CHILD_EXAMPLE_SPEAKER,
    CHILD_METADATA_SCHEMA,
    load_child_chunk_example,
    list_pdf_names,
    load_parent_lookup_summary,
    format_intermediate_outputs,
    parent_text_excerpt,
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

    @patch("streamlit_app.load_parent_lookup", return_value={"parent-id": "A" * 2000})
    @patch("chromadb.PersistentClient")
    def test_loads_one_child_chunk_example_without_embeddings(
        self,
        persistent_client,
        parent_lookup,
    ):
        collection = Mock()
        collection.get.side_effect = [
            {"metadatas": [{
                "parent_id": "parent-id",
                "speaker": CHILD_EXAMPLE_SPEAKER,
            }]},
            {
                "ids": ["child-id"],
                "documents": ["Enriched transcript text"],
                "metadatas": [{
                    "parent_id": "parent-id",
                    "ticker": "INFY",
                    "year": "2025",
                    "month": "07",
                    "index": "1",
                    "speaker": CHILD_EXAMPLE_SPEAKER,
                    "speaker_number": "1",
                    "page_number": "5",
                }],
            },
        ]
        client = persistent_client.return_value
        client.list_collections.return_value = [
            SimpleNamespace(name="infosys_financial_transcripts")
        ]
        client.get_collection.return_value = collection

        example = load_child_chunk_example(Path("project"))

        self.assertEqual(example["id"], "child-id")
        self.assertEqual(example["document"], "Enriched transcript text")
        self.assertEqual(example["metadata"]["parent_id"], "parent-id")
        self.assertEqual(example["metadata"]["speaker"], CHILD_EXAMPLE_SPEAKER)
        self.assertEqual(
            collection.get.call_args_list,
            [
                call(
                    where={"speaker": CHILD_EXAMPLE_SPEAKER},
                    include=["metadatas"],
                ),
                call(
                    where={"parent_id": "parent-id"},
                    limit=1,
                    include=["documents", "metadatas"],
                ),
            ],
        )

    def test_parent_text_excerpt_is_bounded_and_marks_truncation(self):
        self.assertEqual(parent_text_excerpt("short text"), "short text...")
        self.assertEqual(parent_text_excerpt("abcdef", max_length=3), "abc...")
        excerpt = parent_text_excerpt("x" * 1400)
        self.assertEqual(len(excerpt), 1353)
        self.assertTrue(excerpt.endswith("..."))

    def test_submitted_query_uses_existing_graph_and_fixed_portfolio(self):
        swarm = Mock()
        swarm.analyze_market_landscape_with_trace.return_value = (
            [{"node": "query_parser", "state_update": {"time_filter": {}}}],
            "Final report",
        )

        node_updates, report = run_analysis_query(
            "  Analyze performance.  ",
            lambda: swarm,
        )

        self.assertEqual(node_updates[0]["node"], "query_parser")
        self.assertEqual(report, "Final report")
        swarm.analyze_market_landscape_with_trace.assert_called_once_with(
            "Analyze performance.",
            ["INFY"],
        )

    def test_rejects_blank_query_without_creating_graph(self):
        swarm_factory = Mock()

        with self.assertRaisesRegex(ValueError, "Enter a query"):
            run_analysis_query("  ", swarm_factory)

        swarm_factory.assert_not_called()

    def test_formats_node_updates_without_repeating_final_report(self):
        formatted = format_intermediate_outputs([
            {
                "node": "query_parser",
                "state_update": {"time_filter": {"latest": True}},
            },
            {
                "node": "research_analyst",
                "state_update": {
                    "analyst_factual_outputs": {"INFY": "- Revenue grew."},
                    "citation_sources": {"S1": {"year": "2025"}},
                },
            },
            {
                "node": "financial_critic",
                "state_update": {},
            },
        ])

        self.assertIn("=== query_parser ===", formatted)
        self.assertIn("=== research_analyst ===", formatted)
        self.assertIn("citation_sources", formatted)
        self.assertIn("=== financial_critic ===", formatted)
        self.assertIn("Final report is shown in the separate output box below.", formatted)
        self.assertNotIn("- Revenue grew.", formatted.split("=== financial_critic ===")[1])
        self.assertNotIn("final_report", formatted)


if __name__ == "__main__":
    unittest.main()
