import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from core_rag.agents_graph import LangGraphFinancialSwarm
from core_rag.citations import render_citations


class FakeDatabase:
    def query_database_advanced(self, **kwargs):
        return {
            "metadatas": [
                {
                    "parent_id": "parent-1",
                    "year": "2025",
                    "month": "07",
                    "index": "1",
                    "page_number": "5",
                },
                {
                    "parent_id": "parent-2",
                    "year": "2025",
                    "month": "07",
                    "index": "2",
                    "page_number": "8",
                },
            ]
        }


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.content)


class RenderCitationsTests(unittest.TestCase):
    def test_renders_source_metadata_after_each_bullet(self):
        report = "## Findings\n- Revenue grew. [[S1]]"
        sources = {
            "S1": {
                "year": "2025",
                "month": "07",
                "index": "2",
                "page_number": "12",
            }
        }

        self.assertEqual(
            render_citations(report, sources),
            "## Findings\n- Revenue grew. (2025-07, index 2, p. 12)",
        )

    def test_renders_multiple_distinct_sources_for_a_bullet(self):
        report = "- Growth was broad-based. [[S1]] [[S2]]"
        sources = {
            "S1": {"year": "2025", "month": "07", "index": "1", "page_number": "5"},
            "S2": {"year": "2025", "month": "07", "index": "2", "page_number": "8"},
        }

        self.assertEqual(
            render_citations(report, sources),
            "- Growth was broad-based. (2025-07, index 1, p. 5) "
            "(2025-07, index 2, p. 8)",
        )

    def test_rejects_bullets_without_source_ids(self):
        with self.assertRaisesRegex(ValueError, "has no source ID"):
            render_citations("- Unsupported claim.", {})

    def test_rejects_unicode_bullets_without_source_ids(self):
        with self.assertRaisesRegex(ValueError, "has no source ID"):
            render_citations("• Unsupported claim.", {})

    def test_rejects_unknown_source_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown source ID"):
            render_citations("- Claim. [[S9]]", {})

    def test_rejects_incomplete_source_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing metadata field 'page_number'"):
            render_citations(
                "- Claim. [[S1]]",
                {"S1": {"year": "2025", "month": "07", "index": "1"}},
            )


class GraphCitationFlowTests(unittest.TestCase):
    def test_graph_resolves_retrieved_metadata_for_final_bullets(self):
        swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        swarm.db = FakeDatabase()
        swarm.llm = FakeLLM("- Analyst claim. [[S1]]")
        swarm.critic_llm = FakeLLM("## Findings\n- Final claim. [[S1]] [[S2]]")
        swarm._load_parent_text = lambda parent_id: f"Context for {parent_id}"

        initial_state = {
            "user_query": "What changed?",
            "target_tickers": ["INFY"],
            "timeline_keys": {"INFY": ["2025_07"]},
        }
        with redirect_stdout(io.StringIO()):
            analyst_state = swarm.node_research_analyst(initial_state)
            final_state = swarm.node_financial_critic({**initial_state, **analyst_state})

        self.assertIn("[[S1]]", swarm.llm.prompts[0])
        self.assertIn("[[S2]]", swarm.llm.prompts[0])
        self.assertEqual(
            final_state["final_report"],
            "## Findings\n- Final claim. (2025-07, index 1, p. 5) "
            "(2025-07, index 2, p. 8)",
        )


if __name__ == "__main__":
    unittest.main()
