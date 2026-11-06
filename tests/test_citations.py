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
                    "ticker": "INFY",
                    "year": "2025",
                    "month": "07",
                    "index": "1",
                    "speaker_number": "1",
                    "page_number": "5",
                    "speaker": "CEO",
                },
                {
                    "parent_id": "parent-2",
                    "ticker": "INFY",
                    "year": "2025",
                    "month": "07",
                    "index": "2",
                    "speaker_number": "1",
                    "page_number": "8",
                    "speaker": "CEO",
                },
            ]
        }


class FakeLLM:
    def __init__(self, content, is_question=False):
        self.content = content
        self.is_question = is_question
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "Decide whether this transcript parent" in prompt:
            return SimpleNamespace(
                content='{"is_question": ' + str(self.is_question).lower() + '}'
            )
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
                "speaker": "CEO",
            }
        }

        self.assertEqual(
            render_citations(report, sources),
            "## Findings\n- Revenue grew. (2025-07, index 2, p. 12, CEO)",
        )

    def test_renders_multiple_distinct_sources_for_a_bullet(self):
        report = "- Growth was broad-based. [[S1]] [[S2]]"
        sources = {
            "S1": {
                "year": "2025", "month": "07", "index": "1",
                "page_number": "5", "speaker": "CEO",
            },
            "S2": {
                "year": "2025", "month": "07", "index": "2",
                "page_number": "8", "speaker": "CEO",
            },
        }

        self.assertEqual(
            render_citations(report, sources),
            "- Growth was broad-based. (2025-07, index 1, p. 5, CEO) "
            "(2025-07, index 2, p. 8, CEO)",
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
            "time_filter": {
                "all_years": False,
                "latest": False,
                "earliest": False,
                "time_list": ["2025_07"],
            },
            "timeline_keys": {"INFY": ["2025_07"]},
        }
        with redirect_stdout(io.StringIO()):
            analyst_state = swarm.node_research_analyst(initial_state)
            final_state = swarm.node_financial_critic({**initial_state, **analyst_state})

        self.assertIn("[[S1]]", swarm.llm.prompts[-1])
        self.assertIn("[[S2]]", swarm.llm.prompts[-1])
        self.assertEqual(
            final_state["final_report"],
            "## Findings\n- Final claim. (2025-07, index 1, p. 5, CEO) "
            "(2025-07, index 2, p. 8, CEO)",
        )

    def test_graph_includes_answer_parent_with_its_own_citation_metadata(self):
        class QuestionDatabase(FakeDatabase):
            def query_database_advanced(self, **kwargs):
                result = super().query_database_advanced(**kwargs)
                return {"metadatas": [dict(
                    result["metadatas"][0],
                    parent_id="question",
                    speaker_number="3",
                    speaker="Analyst",
                    page_number="10",
                )]}

            def get_parent_by_speaker_number(self, **kwargs):
                self.next_parent_query = kwargs
                return {
                    "parent_id": "answer",
                    "ticker": "INFY",
                    "year": "2025",
                    "month": "07",
                    "index": "1",
                    "speaker_number": "4",
                    "speaker": "CEO",
                    "page_number": "11",
                }

        swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        swarm.db = QuestionDatabase()
        swarm.llm = FakeLLM("- Analyst claim. [[S1]]", is_question=True)
        swarm.critic_llm = FakeLLM("## Findings\n- Final claim. [[S2]]")
        swarm._load_parent_text = lambda parent_id: {
            "question": "Analyst: Could you explain the margin outlook?",
            "answer": "CEO: We expect margins to expand.",
        }[parent_id]
        initial_state = {
            "user_query": "What changed?",
            "target_tickers": ["INFY"],
            "time_filter": {
                "all_years": False,
                "latest": False,
                "earliest": False,
                "time_list": ["2025_07"],
            },
            "timeline_keys": {"INFY": ["2025_07"]},
        }

        with redirect_stdout(io.StringIO()):
            analyst_state = swarm.node_research_analyst(initial_state)
            final_state = swarm.node_financial_critic({**initial_state, **analyst_state})

        self.assertIn("CEO: We expect margins to expand.", swarm.llm.prompts[1])
        self.assertEqual(swarm.db.next_parent_query["speaker_number"], 4)
        self.assertEqual(
            analyst_state["citation_sources"]["S2"],
            {
                "year": "2025",
                "month": "07",
                "index": "1",
                "page_number": "11",
                "speaker": "CEO",
            },
        )
        self.assertEqual(
            final_state["final_report"],
            "## Findings\n- Final claim. (2025-07, index 1, p. 11, CEO)",
        )


if __name__ == "__main__":
    unittest.main()
