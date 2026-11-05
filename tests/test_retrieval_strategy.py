import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from core_rag.agents_graph import LangGraphFinancialSwarm
from core_rag.vector_store import LocalVectorStorage


def child_metadata(ticker="INFY", year="2025", month="07", index="1"):
    return {
        "parent_id": f"{ticker}-{year}-{month}-{index}",
        "ticker": ticker,
        "year": year,
        "month": month,
        "index": index,
        "page_number": "5",
        "speaker": "CEO",
    }


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="- Claim. [[S1]]")


class FakeRetrievalDB:
    def __init__(self):
        self.calls = []
        self.document = {"year": "2025", "month": "07", "index": "2"}

    def query_database(self, **kwargs):
        self.calls.append(("all_years", kwargs))
        metadata = child_metadata(ticker=kwargs["ticker"])
        return {"metadatas": [[metadata]]}

    def query_database_advanced(self, **kwargs):
        self.calls.append(("filtered", kwargs))
        metadata = child_metadata(ticker=kwargs["ticker"])
        return {"metadatas": [metadata]}

    def get_company_extreme_document(self, ticker, latest):
        self.calls.append(("extreme", {"ticker": ticker, "latest": latest}))
        return self.document


class FakeCollection:
    def __init__(self, metadatas=None):
        self.metadatas = metadatas or []
        self.get_calls = []
        self.query_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return {"metadatas": self.metadatas}

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}


class RetrievalStrategyTests(unittest.TestCase):
    def setUp(self):
        self.swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        self.swarm.db = FakeRetrievalDB()
        self.swarm.llm = FakeLLM()
        self.swarm._load_parent_text = lambda parent_id: f"Context for {parent_id}"

    def run_analyst(self, time_filter, tickers=("INFY",)):
        state = {
            "user_query": "What changed?",
            "target_tickers": list(tickers),
            "time_filter": time_filter,
            "timeline_keys": {},
        }
        with redirect_stdout(io.StringIO()):
            return self.swarm.node_research_analyst(state)

    @staticmethod
    def filter(all_years=False, latest=False, earliest=False, time_list=None):
        return {
            "all_years": all_years,
            "latest": latest,
            "earliest": earliest,
            "time_list": time_list or [],
        }

    def test_all_years_uses_unrestricted_ticker_search_with_top_k_five(self):
        self.run_analyst(self.filter(all_years=True, time_list=["2025_07"]))

        self.assertEqual(len(self.swarm.db.calls), 1)
        mode, params = self.swarm.db.calls[0]
        self.assertEqual(mode, "all_years")
        self.assertEqual(params["ticker"], "INFY")
        self.assertEqual(params["top_k"], 5)

    def test_latest_uses_latest_document_and_lowest_index_filter_with_top_k_three(self):
        self.run_analyst(self.filter(latest=True))

        self.assertEqual(self.swarm.db.calls[0], (
            "extreme", {"ticker": "INFY", "latest": True}
        ))
        mode, params = self.swarm.db.calls[1]
        self.assertEqual(mode, "filtered")
        self.assertEqual(params["target_keys"], ["2025_07"])
        self.assertEqual(params["target_index"], "2")
        self.assertEqual(params["top_k"], 3)

    def test_earliest_uses_earliest_document_with_top_k_three(self):
        self.run_analyst(self.filter(earliest=True))

        self.assertEqual(self.swarm.db.calls[0], (
            "extreme", {"ticker": "INFY", "latest": False}
        ))
        self.assertEqual(self.swarm.db.calls[1][1]["top_k"], 3)

    def test_all_years_has_priority_over_other_modes(self):
        self.run_analyst(self.filter(
            all_years=True,
            latest=True,
            earliest=True,
            time_list=["2025_07"]
        ))

        self.assertEqual([call[0] for call in self.swarm.db.calls], ["all_years"])

    def test_latest_has_priority_over_earliest_and_time_list(self):
        self.run_analyst(self.filter(
            latest=True,
            earliest=True,
            time_list=["2025_07"]
        ))

        self.assertEqual([call[0] for call in self.swarm.db.calls], ["extreme", "filtered"])
        self.assertTrue(self.swarm.db.calls[0][1]["latest"])

    def test_time_list_queries_union_of_months_with_top_k_five(self):
        self.run_analyst(self.filter(time_list=["2025_06", "2025_07"]))

        mode, params = self.swarm.db.calls[0]
        self.assertEqual(mode, "filtered")
        self.assertEqual(params["target_keys"], ["2025_06", "2025_07"])
        self.assertEqual(params["top_k"], 5)
        self.assertNotIn("target_index", params)

    def test_selection_runs_independently_for_each_supplied_ticker(self):
        self.run_analyst(self.filter(all_years=True), tickers=("INFY", "TCS"))

        self.assertEqual(
            [call[1]["ticker"] for call in self.swarm.db.calls],
            ["INFY", "TCS"],
        )

    def test_missing_extreme_document_skips_vector_search(self):
        self.swarm.db.document = None

        result = self.run_analyst(self.filter(latest=True))

        self.assertEqual([call[0] for call in self.swarm.db.calls], ["extreme"])
        self.assertEqual(result["analyst_factual_outputs"], {"INFY": ""})

    def test_empty_time_list_without_mode_fails_instead_of_broadening_query(self):
        with self.assertRaisesRegex(ValueError, "empty time_list"):
            self.run_analyst(self.filter())

        self.assertEqual(self.swarm.db.calls, [])

    def test_vector_store_selects_latest_month_then_lowest_index(self):
        collection = FakeCollection([
            {"year": "2024", "month": "12", "index": "1"},
            {"year": "2024", "month": "12", "index": "2"},
            {"year": "2025", "month": "06", "index": "3"},
            {"year": "2025", "month": "06", "index": "1"},
            {"year": "2025", "month": "07", "index": "2"},
            {"year": "invalid", "month": "08", "index": "1"},
            {"year": "2025", "month": "13", "index": "1"},
        ])
        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = collection

        latest = db.get_company_extreme_document("INFY", latest=True)
        earliest = db.get_company_extreme_document("INFY", latest=False)

        self.assertEqual(latest, {"year": "2025", "month": "07", "index": "2"})
        self.assertEqual(earliest, {"year": "2024", "month": "12", "index": "1"})

    def test_vector_store_returns_none_when_no_valid_transcript_metadata_exists(self):
        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = FakeCollection([{"year": "2025", "month": "13", "index": "1"}])

        self.assertIsNone(db.get_company_extreme_document("INFY", latest=True))

    def test_vector_store_filters_query_to_ticker_month_and_index(self):
        collection = FakeCollection()
        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = collection

        db.query_database_advanced(
            query_text="query",
            ticker="INFY",
            target_keys=["2025_07"],
            top_k=3,
            target_index="2"
        )

        self.assertEqual(
            collection.query_calls[0]["where"],
            {
                "$and": [
                    {"ticker": "INFY"},
                    {"$and": [{"year": "2025"}, {"month": "07"}]},
                    {"index": "2"}
                ]
            }
        )

    def test_vector_store_builds_union_filter_for_time_list(self):
        collection = FakeCollection()
        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = collection

        db.query_database_advanced(
            query_text="query",
            ticker="INFY",
            target_keys=["2025_06", "2025_07"],
            top_k=5
        )

        self.assertEqual(
            collection.query_calls[0]["where"],
            {
                "$and": [
                    {"ticker": "INFY"},
                    {"$or": [
                        {"$and": [{"year": "2025"}, {"month": "06"}]},
                        {"$and": [{"year": "2025"}, {"month": "07"}]}
                    ]}
                ]
            }
        )

    def test_vector_store_rejects_non_numeric_target_index(self):
        db = LocalVectorStorage.__new__(LocalVectorStorage)
        db.collection = FakeCollection()

        with self.assertRaisesRegex(ValueError, "expected a numeric index"):
            db.query_database_advanced(
                query_text="query",
                ticker="INFY",
                target_keys=["2025_07"],
                top_k=3,
                target_index="latest"
            )


if __name__ == "__main__":
    unittest.main()
