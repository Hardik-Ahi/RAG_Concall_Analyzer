import os
import json
import re
from datetime import date
from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from core_rag.citations import render_citations
from core_rag.vector_store import LocalVectorStorage


class TimeFilter(TypedDict):
    all_years: bool
    latest: bool
    earliest: bool
    time_list: List[str]


# =====================================================================
# 1. DEFINE THE AGENT STATE MACHINE DATA TYPE
# =====================================================================
class AgentState(TypedDict):
    """
    The permanent memory state shared across all execution nodes in the graph.
    """
    user_query: str
    target_tickers: List[str]
    time_filter: TimeFilter
    timeline_keys: Dict[str, List[str]] # maps ticker -> relevant file keys ["2025_07"]  // generalized for any timeline
    analyst_factual_outputs: Dict[str, str] # maps ticker -> clean facts
    citation_sources: Dict[str, Dict[str, str]]
    final_report: str

# =====================================================================
# 2. THE LANGGRAPH WORKFLOW ENGINES
# =====================================================================
class LangGraphFinancialSwarm:
    def __init__(self, lookup_path: str = "parent_lookup.json"):
        self.db = LocalVectorStorage()
        self.lookup_path = lookup_path
        
        # Instantiate open-source local LangChain wrappers around your Ollama model
        self.llm = ChatOllama(model="llama3.2", temperature=0.0)
        self.critic_llm = ChatOllama(model="llama3.2", temperature=0.3)
        
        # Build the functional graph execution structure
        self.graph = self._compile_agent_graph()
        print("🔗 LangGraph Multi-Agent Architecture successfully compiled and active.")

    def _load_parent_text(self, parent_id: str) -> str:
        if os.path.exists(self.lookup_path):
            with open(self.lookup_path, "r") as f:
                return json.load(f).get(parent_id, "[Extraction Error]")
        return "[Lookup Missing]"

    # --- NODE 0: THE TEMPORAL QUERY ROUTER ---
    def node_query_parser(self, state: AgentState) -> Dict[str, Any]:
        print(f"🧩 [NODE 0: ROUTER] Parsing temporal filters...")

        today = date.today()
        current_date = f"{today.year}-{today.month:02d}"

        prompt = (
            f"Current year and month: {current_date}.\n"
            f"Parse the temporal intent in this user query: {state['user_query']!r}\n"
            "Return ONLY a JSON object with exactly these keys and types, without markdown fences:\n"
            '{"all_years": false, "latest": false, "earliest": false, "time_list": ["2026_06"]}\n'
            "Rules:\n"
            "- all_years is true when the user asks for all items/documents available in the database. "
            "For a query with no temporal constraint, set all_years to true.\n"
            "- latest is true when the user targets the latest available document.\n"
            "- earliest is true when the user targets the earliest available document.\n"
            "- Set each boolean independently according to the user's request. They may overlap; "
            "when the filter is later applied, the consumer gives priority to all_years, then latest, "
            "then earliest.\n"
            "- If none of those modes applies, set all three booleans to false and populate time_list "
            "with every requested calendar month as a zero-padded YYYY_MM string, in chronological "
            "order, without duplicates. time_list must not be empty in this case.\n"
            "- When the query includes an explicit date range or named months, populate time_list "
            "with those months even if one or more booleans are true. If there is no explicit time "
            "range, use an empty time_list.\n"
            "- Expand explicit date ranges inclusively, including both endpoint months. Expand named "
            "months and explicitly listed months to their YYYY_MM values. A year without a month "
            "means all twelve months of that year.\n"
            "- For 'last X months', include the current month and the preceding X-1 calendar months. "
            "For 'last X years', include the current month and the preceding (12*X)-1 calendar months.\n"
            "- Do not infer a narrower period when the user asks for all available data."
        )

        response = self.llm.invoke(prompt)
        clean_response = response.content.strip()
        if clean_response.startswith("```"):
            lines = clean_response.splitlines()
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            clean_response = "\n".join(lines).strip()

        time_filter = json.loads(clean_response)
        expected_keys = {"all_years", "latest", "earliest", "time_list"}
        if not isinstance(time_filter, dict) or set(time_filter) != expected_keys:
            raise ValueError(
                "Temporal filter response must contain exactly all_years, latest, "
                "earliest, and time_list."
            )
        for field in ("all_years", "latest", "earliest"):
            if type(time_filter[field]) is not bool:
                raise ValueError(f"Temporal filter field {field!r} must be a boolean.")

        has_selected_mode = any(
            time_filter[field] for field in ("all_years", "latest", "earliest")
        )

        time_list = time_filter["time_list"]
        if not isinstance(time_list, list):
            raise ValueError("Temporal filter field 'time_list' must be a list.")
        for timestamp in time_list:
            if not isinstance(timestamp, str) or not re.fullmatch(
                r"[0-9]{4}_(?:0[1-9]|1[0-2])", timestamp
            ):
                raise ValueError(
                    f"Invalid time_list entry {timestamp!r}; expected YYYY_MM "
                    "with a valid month."
                )
            if int(timestamp[:4]) < 1:
                raise ValueError(f"Invalid time_list year in {timestamp!r}.")

        if not has_selected_mode and not time_list:
            raise ValueError(
                "Temporal filter time_list must not be empty when all flags are false."
            )
        if len(time_list) != len(set(time_list)):
            raise ValueError("Temporal filter time_list must not contain duplicates.")
        if time_list != sorted(time_list):
            raise ValueError("Temporal filter time_list must be in chronological order.")

        return {"time_filter": time_filter}

    # --- NODE 1: THE RESEARCH ANALYST ---
    def node_research_analyst(self, state: AgentState) -> Dict[str, Any]:
        analyst_outputs = {}
        citation_sources = {}
        source_ids_by_metadata = {}
        
        for ticker in state["target_tickers"]:
            print(f"Crypting 🕵️‍♂️ [NODE 1: ANALYST] Extracting factual profiles for {ticker}...")
            target_keys = state["timeline_keys"].get(ticker, [])
            
            # Advanced chronological vector database fetch
            search_results = self.db.query_database_advanced(  # WHAT ?!
                query_text=state["user_query"], ticker=ticker, target_keys=target_keys, top_k=3
            )
            
            metadatas = search_results.get("metadatas", [])
            parent_contexts = {}
            for meta in metadatas:
                p_id = meta.get("parent_id")
                if not p_id:
                    continue

                citation_metadata = {}
                for field in ("year", "month", "index", "page_number", "speaker"):
                    value = meta.get(field)
                    if value is None or not str(value).strip():
                        raise ValueError(
                            f"Retrieved child metadata for parent {p_id!r} is missing {field!r}."
                        )
                    citation_metadata[field] = str(value).strip()

                source_key = (ticker, *(citation_metadata[field] for field in (
                    "year", "month", "index", "page_number", "speaker"
                )))
                source_id = source_ids_by_metadata.get(source_key)
                if source_id is None:
                    source_id = f"S{len(citation_sources) + 1}"
                    source_ids_by_metadata[source_key] = source_id
                    citation_sources[source_id] = citation_metadata

                if p_id not in parent_contexts:
                    parent_contexts[p_id] = {
                        "text": self._load_parent_text(p_id),
                        "source_ids": []
                    }
                if source_id not in parent_contexts[p_id]["source_ids"]:
                    parent_contexts[p_id]["source_ids"].append(source_id)

            if not parent_contexts:
                analyst_outputs[ticker] = ""
                continue
                    
            context_blocks = []
            for item in parent_contexts.values():
                source_markers = ", ".join(
                    "[[{}]]".format(source_id) for source_id in item["source_ids"]
                )
                context_blocks.append(
                    f"Source IDs: {source_markers}\n{item['text']}"
                )

            compiled_context = "\n\n---\n\n".join(context_blocks)
            
            prompt = (
                "Extract factual claims based ONLY on the evidence below. Return Markdown bullet points. "
                "End every bullet with one or more exact source ID markers shown with that evidence. "
                "Do not invent, alter, or omit source IDs.\n\n"
                f"{compiled_context}\n\n"
                f"Focus on fulfilling this target: {state['user_query']}"
            )
            
            response = self.llm.invoke(prompt)
            analyst_outputs[ticker] = response.content

        return {
            "analyst_factual_outputs": analyst_outputs,
            "citation_sources": citation_sources
        }

    # --- NODE 2: THE FINANCIAL CRITIC ---
    def node_financial_critic(self, state: AgentState) -> Dict[str, Any]:
        print("⚖️ [NODE 2: CRITIC] Executing corporate landscape synthesis analysis...")

        if not state["citation_sources"]:
            return {"final_report": "No source-backed evidence was retrieved for this query."}
        
        raw_evidence = ""
        for ticker, data in state["analyst_factual_outputs"].items():
            raw_evidence += f"=== EVIDENCE BASE FOR {ticker} ===\n{data}\n\n"

        prompt = (
            "You are an expert financial analyst. Use only the source-backed evidence to answer the "
            "user's query. Produce section headings and Markdown bullet points. Every factual bullet "
            "must end with one or more exact source ID markers from the evidence; "
            "preserve those markers unchanged. For combined claims, include every relevant source ID. "
            "Do not make claims that lack a source ID.\n\n"
            f"Evidence:\n{raw_evidence}\n"
            f"User query: {state['user_query']}"
        )
        
        # For LangGraph prototypes running locally, we can capture the complete text output block cleanly
        response = self.critic_llm.invoke(prompt)
        return {
            "final_report": render_citations(
                response.content,
                state["citation_sources"]
            )
        }

    # =====================================================================
    # 3. ASSEMBLING THE COMPILATION GRAPH STATE MACHINE
    # =====================================================================
    def _compile_agent_graph(self):
        # Create the graph blueprint using our typed state schema configuration
        builder = StateGraph(AgentState)
        
        # Register our processing capability nodes
        builder.add_node("query_parser", self.node_query_parser)
        builder.add_node("research_analyst", self.node_research_analyst)
        builder.add_node("financial_critic", self.node_financial_critic)
        
        # Link structural progression steps sequentially using standard edges
        builder.add_edge(START, "query_parser")
        builder.add_edge("query_parser", "research_analyst")
        builder.add_edge("research_analyst", "financial_critic")
        builder.add_edge("financial_critic", END)
        
        return builder.compile()

    def analyze_market_landscape(self, analysis_query: str, default_portfolio: List[str]) -> str:
        """Invokes the LangGraph state machine workflow runtime engine."""
        # Initialize the global state dictionary payload input configurations
        initial_state: AgentState = {
            "user_query": analysis_query,
            "target_tickers": default_portfolio,
            "time_filter": {
                "all_years": False,
                "latest": False,
                "earliest": False,
                "time_list": []
            },
            "timeline_keys": {},
            "analyst_factual_outputs": {},
            "citation_sources": {},
            "final_report": ""
        }
        
        # Run graph instance execution loops
        final_output_state = self.graph.invoke(initial_state)
        return final_output_state["final_report"]
