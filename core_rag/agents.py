import os
import chromadb
import ollama
from vector_store import LocalVectorStorage

class FinancialAgentSwarm:
    def __init__(self):
        # Initialize our local vector storage engine
        self.db = LocalVectorStorage()
        # Verify connection to local running Ollama instance
        print("🤖 Local Multi-Agent Engine Active using Free Llama 3.2 via Ollama.")

    def _agent_research_analyst(self, query: str, ticker: str) -> str:
        """
        Agent 1: The Research Analyst
        Job: Strict factual data gathering locally. Banned from assumptions.
        """
        print(f"🕵️‍♂️ Agent 1 (Research Analyst) is scanning local database records for {ticker}...")
        
        # 1. Fetch vector matches from our local Chroma DB folder
        search_results = self.db.query_database(query_text=query, ticker=ticker, top_k=3)
        documents = search_results.get("documents", [[]])

        context_str = "No direct disclosures found."
        if documents:
            context_str = "\n******\n".join(["\n---\n".join(docs) for docs in documents])

        # 2. Instruct the local model to extract text facts
        system_prompt = (
            "You are a meticulous financial research assistant. Your task is to extract "
            "and list the exact factual claims made by management based ONLY on the context blocks provided. "
            "Do not add personal interpretations or speculation. State your data points clearly as bullet points."
        )
        
        user_prompt = f"User Request: {query}\n\n--- SEARCH CONTEXT BLOCKS ---\n{context_str}"
        
        # Run local generation
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.0} # Lock down creativity to prevent hallucinations
        )
        return response["message"]["content"]

    def _agent_financial_critic(self, analyst_insights: dict) -> str:
        """
        Agent 2: The Financial Critic / Investment Writer
        Job: Local synthesis and structured report formatting.
        """
        print("⚖️ Agent 2 (Financial Critic) is executing comparative intelligence routing locally...")
        
        raw_evidence = ""
        for ticker, data in analyst_insights.items():
            raw_evidence += f"=== FACTUAL DATA FOR {ticker} ===\n{data}\n\n"

        system_prompt = (
            "You are an expert financial analyst. Take the raw data points extracted from earnings calls "
            "and write a sophisticated corporate intelligence briefing.\n\n"
            "Your output MUST follow this strict structured layout:\n"
            "## Executive Synthesis\n"
            "## Cross-Company Comparative Moat Evaluation (skip this if only 1 company is provided)\n"
            "## Key Financial & Forward-Looking Risks\n\n"
            "Maintain a professional, objective tone. Do not use generic buzzwords."
        )
        
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this compiled evidence base:\n\n{raw_evidence}"}
            ],
            options={"temperature": 0.3}
        )
        return response["message"]["content"]

    def analyze_market_landscape(self, analysis_query: str, target_tickers: list[str]) -> str:
        """Coordinates the sequential execution flow between the two local agents."""
        analyst_outputs = {}
        for ticker in target_tickers:
            facts = self._agent_research_analyst(query=analysis_query, ticker=ticker)
            analyst_outputs[ticker] = facts

        final_briefing = self._agent_financial_critic(analyst_outputs)
        return final_briefing

if __name__ == "__main__":
    swarm = FinancialAgentSwarm()
    test_query = "What is management's outlook regarding margins and operational scaling?"
    report = swarm.analyze_market_landscape(analysis_query=test_query, target_tickers=["INFY"])
    
    print("\n================== LOCAL EXECUTIVE REPORT ==================")
    print(report)
    print("=============================================================")
