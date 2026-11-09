import io
import json
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
        self.structured_output_calls = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "Decide whether this transcript parent" in prompt:
            return SimpleNamespace(
                content='{"is_question": ' + str(self.is_question).lower() + '}'
            )
        return SimpleNamespace(content=self.content)

    def with_structured_output(self, schema, method):
        self.structured_output_calls.append((schema, method))
        llm = self

        class StructuredLLM:
            def invoke(self, prompt):
                result = json.loads(llm.invoke(prompt).content)
                return {"summaries": result} if isinstance(result, list) else result

        return StructuredLLM()


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
                "speaker_number": "1",
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
                "page_number": "5", "speaker": "CEO", "speaker_number": "1",
            },
            "S2": {
                "year": "2025", "month": "07", "index": "2",
                "page_number": "8", "speaker": "CEO", "speaker_number": "2",
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
        swarm.llm = FakeLLM(json.dumps(["Revenue grew 10%.", "Margins expanded."]))
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

        self.assertIn("Texts in order:", swarm.llm.prompts[-1])
        self.assertIn("field named 'summaries'", swarm.llm.prompts[-1])
        self.assertIn('"Context for parent-1"', swarm.llm.prompts[-1])
        self.assertIn('"Context for parent-2"', swarm.llm.prompts[-1])
        self.assertEqual(len(swarm.llm.structured_output_calls), 1)
        schema, method = swarm.llm.structured_output_calls[0]
        self.assertEqual(method, "json_schema")
        self.assertEqual(schema["properties"]["summaries"]["items"], {"type": "string"})
        self.assertEqual(schema["properties"]["summaries"]["minItems"], 2)
        self.assertEqual(schema["properties"]["summaries"]["maxItems"], 2)
        self.assertEqual(
            [parent["content"] for parent in analyst_state["parent_chunks"]],
            ["Revenue grew 10%.", "Margins expanded."],
        )
        self.assertEqual(
            [parent["citation_number"] for parent in analyst_state["parent_chunks"]],
            ["S1", "S2"],
        )
        self.assertEqual(analyst_state["citation_sources"]["S1"]["parent_id"], "parent-1")
        self.assertEqual(analyst_state["citation_sources"]["S2"]["parent_id"], "parent-2")
        self.assertIn("Summarized parent records:", swarm.critic_llm.prompts[0])
        self.assertIn('"citation_number": "S1"', swarm.critic_llm.prompts[0])
        self.assertNotIn('"parent_id"', swarm.critic_llm.prompts[0])
        self.assertIn("| Citation | Ticker | Year | Month | Index | Page | Speaker | Speaker number |", final_state["final_report"])
        self.assertNotIn("parent-1", final_state["final_report"])
        self.assertEqual(
            final_state["final_report"],
            "## Findings\n- Final claim. (2025-07, index 1, p. 5, CEO) "
            "(2025-07, index 2, p. 8, CEO)\n\n"
            "## Citation mapping\n\n"
            "| Citation | Ticker | Year | Month | Index | Page | Speaker | Speaker number |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| S1 | INFY | 2025 | 07 | 1 | 5 | CEO | 1 |\n"
            "| S2 | INFY | 2025 | 07 | 2 | 8 | CEO | 1 |",
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
        swarm.llm = FakeLLM(
            json.dumps(["Question summary.", "Margins will expand."]),
            is_question=True,
        )
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
                "parent_id": "answer",
                "ticker": "INFY",
                "year": "2025",
                "month": "07",
                "index": "1",
                "page_number": "11",
                "speaker": "CEO",
                "speaker_number": "4",
            },
        )
        self.assertEqual(analyst_state["parent_chunks"][1]["content"], "Margins will expand.")
        self.assertIn(
            "| S2 | INFY | 2025 | 07 | 1 | 11 | CEO | 4 |",
            final_state["final_report"],
        )

    def test_empty_analyst_summary_keeps_position_but_has_no_citation(self):
        swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        swarm.db = FakeDatabase()
        swarm.llm = FakeLLM(json.dumps(["", "Revenue grew 10%."]))
        swarm.critic_llm = FakeLLM("## Findings\n- Revenue grew 10%. [[S1]]")
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

        self.assertEqual(
            [parent["citation_number"] for parent in analyst_state["parent_chunks"]],
            [None, "S1"],
        )
        self.assertEqual(list(analyst_state["citation_sources"]), ["S1"])
        self.assertNotIn("parent-1", swarm.critic_llm.prompts[0])
        self.assertNotIn("parent-1", final_state["final_report"])

    def test_rejects_analyst_output_with_wrong_number_of_summaries(self):
        swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        swarm.db = FakeDatabase()
        swarm.llm = FakeLLM(json.dumps(["Only one summary."]))
        swarm._load_parent_text = lambda parent_id: f"Context for {parent_id}"
        state = {
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

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            ValueError, "exactly 2 strings in parent order"
        ):
            swarm.node_research_analyst(state)


if __name__ == "__main__":
    unittest.main()
