import os
import json
import ollama
from vector_store import LocalVectorStorage

class FinancialAgentSwarm:
    def __init__(self, lookup_path: str = "parent_lookup.json"):
        self.db = LocalVectorStorage()
        self.lookup_path = lookup_path
        print("🤖 Local Multi-Agent Engine Active using Relational Parent-Child Extraction.")

    def _load_parent_text(self, parent_id: str) -> str:
        """Helper to resolve a child's parent_id into the original full paragraph speech text."""
        if os.path.exists(self.lookup_path):
            with open(self.lookup_path, "r") as f:
                lookup = json.load(f)
                return lookup.get(parent_id, "[Parent Text Recovery Error]")
        return "[Lookup Database Missing]"

    def _agent_research_analyst(self, query: str, ticker: str) -> str:
        """
        Agent 1: The Research Analyst
        Performs vector queries, executes relational context switches from child to parent,
        and extracts strict factual blocks.
        """
        print(f"\n🕵️ [AGENT 1] Scanning vectors for {ticker}...")
        
        # 1. Fetch top child semantic hits
        search_results = self.db.query_database(query_text=query, ticker=ticker, top_k=2)
        
        child_docs = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        
        # --- TRANSPARENT PRINT: VISUALIZE CHILD CHUNKS RETRIEVED ---
        print(f"\n======== 📥 STEP 1: CHILD CHUNKS RETRIEVED ({ticker}) ========")
        for idx, child_text in enumerate(child_docs):
            print(f"\n[Child Match #{idx+1} | ID: {search_results['ids'][0][idx][:8]}...]")
            print(f"Text Snippet:\n{child_text[:200]}...")
        print("==============================================================")

        # 2. Relational context mapping step: Resolve children to parent blocks
        parent_contexts = set()
        for meta in metadatas:
            p_id = meta.get("parent_id")
            if p_id:
                parent_text = self._load_parent_text(p_id)
                parent_contexts.add(parent_text)

        # Remove duplicate parent contexts if multiple child matches hit the same speech block
        compiled_parent_context_str = "\n\n---\n\n".join(list(parent_contexts))

        # --- TRANSPARENT PRINT: VISUALIZE PARENT CONTEXT SENT TO LLM ---
        print(f"\n======== 🏛️ STEP 2: PARENT CHUNKS PASSED TO LLM ({ticker}) ========")
        print(compiled_parent_context_str[:1200] + "\n... [Truncated for Console View]")
        print("================================================================")

        # 3. Instruct Llama 3.2 using the massive Parent Context Block
        system_prompt = (
            "You are a meticulous financial research assistant. Your task is to extract "
            "and list the exact factual claims made by management based ONLY on the context blocks provided. "
            "The context provided represents the broader conversations surrounding relevant disclosures. "
            "State your data points clearly as explicit bullet points."
        )
        
        user_prompt = f"User Request: {query}\n\n--- BROADER PARENT CONTEXT --- \n{compiled_parent_context_str}"
        
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0}
        )
        return response["message"]["content"]

    def _agent_financial_critic(self, analyst_insights: dict) -> str:
        """Agent 2: Synthesizes the final corporate intelligence brief."""
        print("\n⚖️ [AGENT 2] Executing comparative analysis across extracted insights...")
        
        raw_evidence = ""
        for ticker, data in analyst_insights.items():
            raw_evidence += f"=== FACTUAL INSIGHTS FOR {ticker} ===\n{data}\n\n"

        system_prompt = (
            "You are an expert financial analyst. Take the raw factual data points extracted from "
            "earnings calls of a stock ticker and compile a sophisticated corporate intelligence briefing.\n\n"
            "Your output MUST follow this strict structured layout:\n"
            "## Executive Synthesis\n"
            "## Cross-Company Comparative Moat Evaluation\n"
            "## Key Financial & Forward-Looking Risks\n\n"
            "Maintain a professional, objective tone."
        )
        
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this evidence base:\n\n{raw_evidence}"}
            ],
            options={"temperature": 0.3}
        )
        return response["message"]["content"]

    def analyze_market_landscape(self, analysis_query: str, target_tickers: list[str]) -> str:
        analyst_outputs = {}
        for ticker in target_tickers:
            facts = self._agent_research_analyst(query=analysis_query, ticker=ticker)
            analyst_outputs[ticker] = facts

        final_briefing = self._agent_financial_critic(analyst_outputs)
        return final_briefing

if __name__ == "__main__":
    swarm = FinancialAgentSwarm()
    report = swarm.analyze_market_landscape("What is management's outlook on margins and AI?", ["INFY"])
