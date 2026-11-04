import os
import json
from datetime import date
from typing import Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from core_rag.vector_store import LocalVectorStorage

# =====================================================================
# 1. DEFINE THE AGENT STATE MACHINE DATA TYPE
# =====================================================================
class AgentState(TypedDict):
    """
    The permanent memory state shared across all execution nodes in the graph.
    """
    user_query: str
    target_tickers: List[str]
    timeline_keys: Dict[str, List[str]] # maps ticker -> relevant file keys ["2025_07"]  // generalized for any timeline
    analyst_factual_outputs: Dict[str, str] # maps ticker -> clean facts
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
        print(f"🧩 [NODE 0: ROUTER] Parsing time constraints and timeline logs...")

        today = date.today()
        current_year = today.year
        current_month = today.month
        
        prompt = (
            f"PRESENT DATE: Year-{current_year}, Month-{current_month:02d}. Extract date filters from this query: '{state['user_query']}'\n"
            "Return ONLY a raw valid JSON payload with this exact structure (no markdown wrappers):\n"
            "{\"ticker\": \"INFY\", \"relative_count\": 4, \"explicit_years\": null}"
        )
        
        # Invoke the local model
        response = self.llm.invoke(prompt)
        
        try:
            clean_str = response.content.strip().replace("```json", "").replace("```", "")
            time_meta = json.loads(clean_str)
        except Exception:
            time_meta = {"ticker": None, "relative_count": 3}

        # Resolve which tickers are active
        ticker_target = time_meta.get("ticker")
        active_tickers = [ticker_target] if ticker_target and ticker_target != "null" else state["target_tickers"]
        relative_count = time_meta.get("relative_count") or 3

        # Map dynamic chronological file footprints for each ticker
        timeline_keys = {}
        for ticker in active_tickers:
            full_timeline = self.db.get_company_timeline(ticker)  # WHAT?!
            if full_timeline:
                sliced = full_timeline[:int(relative_count)]
                timeline_keys[ticker] = [item["key"] for item in sliced]

        return {"timeline_keys": timeline_keys, "target_tickers": active_tickers}

    # --- NODE 1: THE RESEARCH ANALYST ---
    def node_research_analyst(self, state: AgentState) -> Dict[str, Any]:
        analyst_outputs = {}
        
        for ticker in state["target_tickers"]:
            print(f"Crypting 🕵️‍♂️ [NODE 1: ANALYST] Extracting factual profiles for {ticker}...")
            target_keys = state["timeline_keys"].get(ticker, [])
            
            # Advanced chronological vector database fetch
            search_results = self.db.query_database_advanced(  # WHAT ?!
                query_text=state["user_query"], ticker=ticker, target_keys=target_keys, top_k=3
            )
            
            metadatas = search_results.get("metadatas", [[]])
            parent_contexts = []
            for meta in metadatas:
                p_id = meta.get("parent_id")
                if p_id:
                    parent_contexts.append(self._load_parent_text(p_id))
                    
            compiled_context = "\n\n---\n\n".join(list(set(parent_contexts)))
            
            prompt = (
                f"Extract clean factual bullet claims based ONLY on this text:\n{compiled_context}\n\n"
                f"Focus on fulfilling this target: {state['user_query']}"
            )
            
            response = self.llm.invoke(prompt)
            analyst_outputs[ticker] = response.content

        return {"analyst_factual_outputs": analyst_outputs}

    # --- NODE 2: THE FINANCIAL CRITIC ---
    def node_financial_critic(self, state: AgentState) -> Dict[str, Any]:
        print("⚖️ [NODE 2: CRITIC] Executing corporate landscape synthesis analysis...")
        
        raw_evidence = ""
        for ticker, data in state["analyst_factual_outputs"].items():
            raw_evidence += f"=== EVIDENCE BASE FOR {ticker} ===\n{data}\n\n"

        prompt = (
            "You are an expert investment analyst. Take this raw data and compile a sophisticated briefing ONLY based on the raw data.\n\n"
            "Your output MUST follow this strict structured layout:\n"
            "## Executive Synthesis\n"
            "## Cross-Company Comparative Moat Evaluation (if applicable on raw data)\n"
            "## Key Financial & Forward-Looking Risks\n\n"
            f"Evidence Context:\n{raw_evidence}"
        )
        
        # For LangGraph prototypes running locally, we can capture the complete text output block cleanly
        response = self.critic_llm.invoke(prompt)
        return {"final_report": response.content}

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
            "timeline_keys": {},
            "analyst_factual_outputs": {},
            "final_report": ""
        }
        
        # Run graph instance execution loops
        final_output_state = self.graph.invoke(initial_state)
        return final_output_state["final_report"]
