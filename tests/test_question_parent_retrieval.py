import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from core_rag.agents import FinancialAgentSwarm
from core_rag.chunker import HierarchicalTranscriptChunker
from core_rag.retrieval import expand_question_parent_hits
from core_rag.vector_store import LocalVectorStorage


class FakeDatabase:
    def __init__(self, next_parent=None):
        self.next_parent = next_parent
        self.calls = []

    def get_parent_by_speaker_number(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_parent


class QuestionParentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.question_metadata = {
            "parent_id": "question-parent",
            "ticker": "INFY",
            "year": "2025",
            "month": "07",
            "index": "2",
            "speaker_number": "8",
            "speaker": "Analyst",
            "page_number": "12",
        }
        self.answer_metadata = {
            "parent_id": "answer-parent",
            "ticker": "INFY",
            "year": "2025",
            "month": "07",
            "index": "2",
            "speaker_number": "9",
            "speaker": "CEO",
            "page_number": "13",
        }
        self.texts = {
            "question-parent": "Analyst: What drove the cost increase?",
            "answer-parent": "CEO: The increase was due to freight costs.",
        }

    def expand(self, db, response, hits=None):
        prompts = []
        results = expand_question_parent_hits(
            parent_hits=hits or [{"metadata": self.question_metadata}],
            db=db,
            load_parent_text=self.texts.__getitem__,
            get_classifier_response=lambda prompt: prompts.append(prompt) or response
        )
        return results, prompts

    def test_non_question_parent_is_returned_without_next_speaker_lookup(self):
        db = FakeDatabase(self.answer_metadata)

        parents, _ = self.expand(db, '{"is_question": false}')

        self.assertEqual([parent["parent_id"] for parent in parents], ["question-parent"])
        self.assertEqual(db.calls, [])

    def test_question_parent_includes_next_speaker_and_scopes_lookup(self):
        db = FakeDatabase(self.answer_metadata)
        duplicate_hit = {"metadata": dict(self.question_metadata)}

        parents, prompts = self.expand(
            db,
            '{"is_question": true}',
            hits=[{"metadata": self.question_metadata}, duplicate_hit]
        )

        self.assertEqual(
            [parent["parent_id"] for parent in parents],
            ["question-parent", "answer-parent"]
        )
        self.assertEqual(parents[1]["metadata"], self.answer_metadata)
        self.assertEqual(
            db.calls,
            [{
                "ticker": "INFY",
                "year": "2025",
                "month": "07",
                "index": "2",
                "speaker_number": 9,
            }]
        )
        self.assertEqual(len(prompts), 1)

    def test_question_without_next_parent_retains_original_context(self):
        parents, _ = self.expand(FakeDatabase(), '{"is_question": true}')

        self.assertEqual([parent["parent_id"] for parent in parents], ["question-parent"])

    def test_invalid_classifier_response_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "invalid result"):
            self.expand(FakeDatabase(), '{"is_question": "yes"}')

    def test_question_without_speaker_number_requires_reindexing(self):
        metadata = dict(self.question_metadata)
        metadata.pop("speaker_number")

        with self.assertRaisesRegex(ValueError, "Reindex transcript chunks"):
            self.expand(
                FakeDatabase(),
                '{"is_question": true}',
                hits=[{"metadata": metadata}]
            )

    def test_short_filtered_blocks_do_not_create_speaker_number_gaps(self):
        chunker = HierarchicalTranscriptChunker()
        with redirect_stdout(StringIO()):
            parents = chunker.build_parent_chunks([{
                "source_file": "INFY_Concall_2025_07_2.pdf",
                "page_number": 1,
                "content": (
                    "Jane Doe\n"
                    "Short turn\n"
                    "John Smith\n"
                    "This longer speaker turn is retained as a complete parent chunk."
                ),
            }])

        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["speaker"], "John Smith")
        self.assertEqual(parents[0]["speaker_number"], 1)

    def test_speaker_number_is_persisted_in_chroma_metadata(self):
        class Collection:
            def add(self, **kwargs):
                self.metadatas = kwargs["metadatas"]

        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.db_dir = "test"
        db.collection = Collection()
        with redirect_stdout(StringIO()):
            db.index_child_chunks([{
                "child_id": "child",
                "parent_id": "parent",
                "ticker": "INFY",
                "year": "2025",
                "month": "07",
                "index": "2",
                "speaker": "CEO",
                "speaker_number": 9,
                "page_number": 13,
                "enriched_text": "text",
            }])

        self.assertEqual(db.collection.metadatas[0]["speaker_number"], "9")

    def test_next_parent_lookup_filters_on_transcript_and_speaker_number(self):
        class Collection:
            def get(self, **kwargs):
                self.kwargs = kwargs
                return {"metadatas": [self.metadata]}

            metadata = dict(self.answer_metadata)

        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = Collection()

        metadata = db.get_parent_by_speaker_number(
            ticker="INFY",
            year="2025",
            month="07",
            index="2",
            speaker_number=9
        )

        self.assertEqual(metadata, self.answer_metadata)
        self.assertEqual(
            db.collection.kwargs["where"],
            {
                "$and": [
                    {"ticker": "INFY"},
                    {"year": "2025"},
                    {"month": "07"},
                    {"index": "2"},
                    {"speaker_number": "9"},
                ]
            }
        )

    def test_legacy_agent_also_includes_the_next_speaker_parent(self):
        class RetrievalDB(FakeDatabase):
            def query_database(self, **kwargs):
                return {
                    "ids": [["child-id"]],
                    "documents": [["question child"]],
                    "metadatas": [[self.question_metadata]],
                }

        retrieval_db = RetrievalDB(self.answer_metadata)
        retrieval_db.question_metadata = self.question_metadata
        agent = FinancialAgentSwarm.__new__(FinancialAgentSwarm)
        agent.db = retrieval_db
        agent.lookup_path = "unused"
        agent._load_parent_text = self.texts.__getitem__
        chat_responses = [
            {"message": {"content": '{"is_question": true}'}},
            {"message": {"content": "Extracted facts"}},
        ]

        with patch("core_rag.agents.ollama.chat", side_effect=chat_responses) as chat:
            with redirect_stdout(StringIO()):
                result = agent._agent_research_analyst("What happened?", "INFY")

        self.assertEqual(result, "Extracted facts")
        self.assertEqual(retrieval_db.calls[0]["speaker_number"], 9)
        self.assertIn("CEO: The increase was due to freight costs.", chat.call_args_list[1].kwargs["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
