import unittest
from unittest.mock import Mock

from core_rag.agents_graph import LangGraphFinancialSwarm


class GraphTraceTests(unittest.TestCase):
    def setUp(self):
        self.swarm = LangGraphFinancialSwarm.__new__(LangGraphFinancialSwarm)
        self.swarm.graph = Mock()
        self.swarm.graph.stream.return_value = iter([
            {"query_parser": {"time_filter": {"latest": True}}},
            {
                "research_analyst": {
                    "analyst_factual_outputs": {"INFY": "- Revenue grew."},
                    "citation_sources": {"S1": {"year": "2025"}},
                }
            },
            {"financial_critic": {"final_report": "Final answer."}},
        ])

    def test_streams_node_state_updates_separately_from_final_report(self):
        node_updates, final_report = self.swarm.analyze_market_landscape_with_trace(
            "Analyze performance.",
            ["INFY"],
        )

        self.assertEqual(
            [update["node"] for update in node_updates],
            ["query_parser", "research_analyst", "financial_critic"],
        )
        self.assertEqual(
            node_updates[0]["state_update"],
            {"time_filter": {"latest": True}},
        )
        self.assertEqual(
            node_updates[1]["state_update"],
            {
                "analyst_factual_outputs": {"INFY": "- Revenue grew."},
                "citation_sources": {"S1": {"year": "2025"}},
            },
        )
        self.assertEqual(node_updates[2]["state_update"], {})
        self.assertEqual(final_report, "Final answer.")
        initial_state = self.swarm.graph.stream.call_args.args[0]
        self.assertEqual(
            self.swarm.graph.stream.call_args.kwargs["stream_mode"],
            "updates",
        )
        self.assertEqual(initial_state["user_query"], "Analyze performance.")
        self.assertEqual(initial_state["target_tickers"], ["INFY"])

    def test_existing_method_returns_only_final_report(self):
        self.swarm.analyze_market_landscape_with_trace = Mock(
            return_value=([{"node": "financial_critic", "state_update": {}}], "Final answer.")
        )

        result = self.swarm.analyze_market_landscape("Analyze performance.", ["INFY"])

        self.assertEqual(result, "Final answer.")


if __name__ == "__main__":
    unittest.main()
