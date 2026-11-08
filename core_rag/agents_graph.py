import os
import json
import re
from datetime import date
from typing import Dict, Any, Iterator, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from core_rag.citations import render_citations
from core_rag.vector_store import LocalVectorStorage
from core_rag.retrieval import expand_question_parent_hits

LLM_MODEL = "llama3.2"
CRITIC_LLM_MODEL = "llama3.2"


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
    def __init__(
        self,
        lookup_path: str = "parent_lookup.json",
        db_dir: str = "chroma_db"
    ):
        self.db = LocalVectorStorage(db_dir=db_dir)
        self.lookup_path = lookup_path
        
        # Instantiate open-source local LangChain wrappers around your Ollama model
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.0)
        self.critic_llm = ChatOllama(model=CRITIC_LLM_MODEL, temperature=0.3)
        
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
            '{"all_years": Boolean, "latest": Boolean, "earliest": Boolean}\n'
            "Rules:\n"
            "- 'all_years' is true only when the user does not provide a temporal constraint or explicitly asks for all data with the keyword 'all' beside the time unit.\n"
            "- 'latest' is true when the user mentions the keywords 'latest' or 'newest' for the data.\n"
            "- 'earliest' is true when the user mentions the keywords 'earliest' or 'oldest' for the data.\n"
        )

        response = self.llm.invoke(prompt)

        def parse_json_response(content: str) -> Any:
            clean_content = content.strip()
            if clean_content.startswith("```"):
                lines = clean_content.splitlines()
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                clean_content = "\n".join(lines).strip()
            return json.loads(clean_content)

        mode_filter = parse_json_response(response.content)
        expected_mode_keys = {"all_years", "latest", "earliest"}
        if not isinstance(mode_filter, dict) or set(mode_filter) != expected_mode_keys:
            raise ValueError(
                "Temporal mode response must contain exactly all_years, latest, and earliest."
            )
        for field in ("all_years", "latest", "earliest"):
            if type(mode_filter[field]) is not bool:
                raise ValueError(f"Temporal filter field {field!r} must be a boolean.")

        if any(mode_filter.values()):
            time_filter = {**mode_filter, "time_list": []}
        else:
            time_list_prompt = (
                f"Current year and month: {current_date}.\n"
                f"Parse the requested time period from this user query: {state['user_query']!r}\n"
                "Return ONLY a JSON array of strings, with every requested month represented as 'YYYY_MM'.\n"
                "Rules:\n"
                "- Convert each explicitly named or listed month in the user query to its corresponding YYYY_MM value.\n"
                "- If the query specifies a year without a month, include all 12 months of that year.\n"
                "- Do not return an empty array. Do not write anything else besides the JSON array. Do not include markdown fences.\n"
            )
            time_list_response = self.llm.invoke(time_list_prompt)
            print(f"🧩 [NODE 0: ROUTER] Time List response: {time_list_response.content}")
            time_list = parse_json_response(time_list_response.content)
            time_filter = {**mode_filter, "time_list": time_list}

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
        '''
        if time_list != sorted(time_list):
            raise ValueError("Temporal filter time_list must be in chronological order.")
        '''

        return {"time_filter": time_filter}

    # --- NODE 1: THE RESEARCH ANALYST ---
    def node_research_analyst(self, state: AgentState) -> Dict[str, Any]:
        analyst_outputs = {}
        citation_sources = {}
        source_ids_by_metadata = {}
        
        for ticker in state["target_tickers"]:
            print(f"Crypting 🕵️‍♂️ [NODE 1: ANALYST] Extracting factual profiles for {ticker}...")
            time_filter = state["time_filter"]
            if time_filter["all_years"]:
                search_results = self.db.query_database(
                    query_text=state["user_query"],
                    ticker=ticker,
                    top_k=5
                )
            elif time_filter["latest"] or time_filter["earliest"]:
                latest = time_filter["latest"]
                document = self.db.get_company_extreme_document(
                    ticker=ticker,
                    latest=latest
                )
                if document is None:
                    search_results = {"metadatas": []}
                else:
                    search_results = self.db.query_database_advanced(
                        query_text=state["user_query"],
                        ticker=ticker,
                        target_keys=[f"{document['year']}_{document['month']}"],
                        target_index=document["index"],
                        top_k=3
                    )
            else:
                if not time_filter["time_list"]:
                    raise ValueError(
                        f"Temporal filter for ticker {ticker!r} has no selected mode "
                        "and an empty time_list."
                    )
                search_results = self.db.query_database_advanced(
                    query_text=state["user_query"],
                    ticker=ticker,
                    target_keys=time_filter["time_list"],
                    top_k=5
                )

            metadatas = search_results.get("metadatas") or []
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            parent_hits = [
                {"metadata": meta}
                for meta in metadatas
                if meta and meta.get("parent_id")
            ]
            expanded_parents = expand_question_parent_hits(
                parent_hits=parent_hits,
                db=self.db,
                load_parent_text=self._load_parent_text,
                get_classifier_response=lambda prompt: self.llm.invoke(prompt).content
            )
            parent_contexts = {}
            for parent in expanded_parents:
                meta = parent["metadata"]
                p_id = parent["parent_id"]

                citation_metadata = {}
                for field in ("year", "month", "index", "page_number", "speaker", "speaker_number"):
                    value = meta.get(field)
                    if value is None or not str(value).strip():
                        raise ValueError(
                            f"Retrieved child metadata for parent {p_id!r} is missing {field!r}."
                        )
                    citation_metadata[field] = str(value).strip()

                source_key = (ticker, *(citation_metadata[field] for field in (
                    "year", "month", "index", "page_number", "speaker", "speaker_number"
                )))
                source_id = source_ids_by_metadata.get(source_key)
                if source_id is None:
                    source_id = f"S{len(citation_sources) + 1}"
                    source_ids_by_metadata[source_key] = source_id
                    citation_sources[source_id] = citation_metadata

                if p_id not in parent_contexts:
                    parent_contexts[p_id] = {
                        "text": parent["text"],
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
            "You are an expert financial analyst. Use the provided evidence to answer the "
            "user's query. Produce Markdown bullet points and a final summary."
            "Map a source ID marker to each provided citation source (which is a Python dictionary) in the form S1, S2, etc. "
            "Write your entire response with the singular aim of covering each source ID marker in your answer."
            "End every bullet with corresponding source ID markers like [S1], [S2] etc. A bullet can have one or more markers. "
            "Do not make claims that lack a source ID marker. And at the end, list the mapping of all source ID markers (S1, S2, etc.) to original citation dictionaries.\n\n"
            f"Evidence:\n{raw_evidence}\n"
            f"User query: {state['user_query']}\n"
            f"Citation sources: {json.dumps(state['citation_sources'], indent=4)}\n"
        )
        
        # For LangGraph prototypes running locally, we can capture the complete text output block cleanly
        response = self.critic_llm.invoke(prompt)
        return {"final_report": response.content}

        '''{
            "final_report": render_citations(
                response.content,
                state["citation_sources"]
            )
        }'''

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

    def analyze_market_landscape_with_trace(
        self,
        analysis_query: str,
        default_portfolio: List[str]
    ) -> tuple[List[Dict[str, Any]], str]:
        """Run the graph and return node state updates separately from its final report."""
        node_updates = []
        final_report = None
        for event in self.stream_market_landscape(analysis_query, default_portfolio):
            if "final_report" in event:
                final_report = event["final_report"]
            else:
                node_updates.append(event)

        if final_report is None:
            raise RuntimeError("Financial critic graph node did not produce a final report.")
        return node_updates, final_report

    def stream_market_landscape(
        self,
        analysis_query: str,
        default_portfolio: List[str]
    ) -> Iterator[Dict[str, Any]]:
        """Yield each completed node's state delta before the graph finishes."""
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

        final_report = None
        for update in self.graph.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in update.items():
                if not isinstance(state_update, dict):
                    raise ValueError(
                        f"Graph node {node_name!r} returned a non-dictionary state update."
                    )
                final_report_update = state_update.get("final_report")
                if final_report_update is not None:
                    final_report = final_report_update
                intermediate_update = {
                    key: value
                    for key, value in state_update.items()
                    if key != "final_report"
                }
                yield {
                    "node": node_name,
                    "state_update": intermediate_update,
                }

        if final_report is None:
            raise RuntimeError("Financial critic graph node did not produce a final report.")
        if not isinstance(final_report, str):
            raise ValueError("Financial critic final report must be a string.")
        yield {"final_report": final_report}

    def analyze_market_landscape(self, analysis_query: str, default_portfolio: List[str]) -> str:
        """Invoke the graph and return only its final report, preserving the CLI API."""
        _, final_report = self.analyze_market_landscape_with_trace(
            analysis_query,
            default_portfolio
        )
        return final_report
