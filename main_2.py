import os
import json
from pathlib import Path

# Explicit imports from your library modules
from data_scripts.downloader import ConcallDownloader
from core_rag.pdf_parser import FinancialPDFParser
from core_rag.chunker import HierarchicalTranscriptChunker
from core_rag.vector_store import LocalVectorStorage
from core_rag.agents import FinancialAgentSwarm

# Global system configuration constants
LOOKUP_PATH = "parent_lookup.json"
COLLECTION_NAME = "concall_vectors"

# =====================================================================
# FUNCTIONAL PIPELINE SUBSYSTEMS (Isolated Capabilities)
# =====================================================================

def run_bulk_ingestion(portfolio: list[str]):
    """
    PHASE 1: Web Scraping & Asset Acquisition
    Programmatically extracts all listed transcripts from Screener.in for the target companies.
    """
    print("\n==============================================================")
    print("📡 PHASE 1: WEB DISCOVERY & BULK DOWNLOADING")
    print("==============================================================")
    downloader = ConcallDownloader()
    
    for ticker in portfolio:
        downloader.fetch_all_transcripts(ticker)
        
    print("✅ Ingestion check complete.")


def run_parsing_and_hierarchical_chunking(portfolio: list[str]) -> tuple[list[dict], list[dict]]:
    """
    PHASE 2 & 3: ETL, Text Layout Parsing, and Speaker-Turn Chunking
    Gathers local PDFs, resolves speaker boundaries, and generates parent-child structures.
    """
    print("\n==============================================================")
    print("📖 PHASE 2 & 3: TEXT PARSING & HIERARCHICAL SEGMENTATION")
    print("==============================================================")
    parser = FinancialPDFParser()
    chunker = HierarchicalTranscriptChunker()
    
    # Track paths of files matching our timestamp criteria recursively across tickers
    transcript_root = Path("financial_data")
    all_pdf_paths = []
    
    for ticker in portfolio:
        company_dir = transcript_root / ticker
        if company_dir.exists():
            pdf_paths = list(company_dir.glob("*_Concall_*.pdf"))
            all_pdf_paths.extend(pdf_paths)

    if not all_pdf_paths:
        print("⚠️ Warning: No date-stamped transcript PDFs found to parse on disk.")
        return [], []

    print(f"📚 Detected {len(all_pdf_paths)} document targets. Beginning structural parsing...")
    
    compiled_parents = []
    compiled_children = []

    for pdf_path in all_pdf_paths:
        # Extract chronological lines page-by-page
        pages = parser.extract_clean_text(str(pdf_path))
        if not pages:
            continue
        
        # Build Parent blocks (UUIDs mapped to speaker dialogue turns)
        file_parents = chunker.build_parent_chunks(pages)
        # Build Child blocks (Sliding window token structures with context enrichment headers)
        file_children = chunker.build_child_chunks(file_parents)
        
        compiled_parents.extend(file_parents)
        compiled_children.extend(file_children)

    # --- ATOMIC METADATA LOOKUP LAYER SYNC ---
    # Construct/update our central JSON lookup map so Agent 2 can resolve parent IDs instantly
    lookup_dict = {p["parent_id"]: p["parent_text"] for p in compiled_parents}
    
    if os.path.exists(LOOKUP_PATH):
        try:
            with open(LOOKUP_PATH, "r") as f:
                existing_data = json.load(f)
                lookup_dict.update(existing_data)
        except Exception:
            pass

    with open(LOOKUP_PATH, "w") as f:
        json.dump(lookup_dict, f, indent=4)
        
    print(f"💾 Relational map synchronized. {len(lookup_dict)} unique parent IDs committed to disk.")
    return compiled_parents, compiled_children


def run_vector_store_hydration(child_chunks: list[dict]):
    """
    PHASE 4: Local Vectorization & Database Storage
    Wipes the active Chroma collection and rebuilds it using local BGE embeddings.
    """
    print("\n==============================================================")
    print("🧠 PHASE 4: LOCAL VECTOR STORE GENERATION & HYDRATION")
    print("==============================================================")
    if not child_chunks:
        print("❌ Sync aborted: 0 child chunks passed for vector database injection.")
        return

    db = LocalVectorStorage()
    
    # Purge old collection to prevent vector fragmentation across script restarts
    try:
        print(f"🧹 Clearing historical records from collection: '{COLLECTION_NAME}'...")
        db.client.delete_collection(COLLECTION_NAME)
        db.collection = db.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=db.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception:
        pass

    # Batch execute embedding computations and save directly to disk database folder
    db.index_child_chunks(child_chunks)
    print("✅ Collection successfully populated.")


def execute_multi_agent_analysis(portfolio: list[str], analysis_query: str):
    """
    PHASE 5: Collaborative Agent Routing Layer
    Deploys the local Llama model via Ollama to evaluate findings across target companies.
    """
    print("\n==============================================================")
    print("⚖️ PHASE 5: RUNNING DUAL-AGENT INTELLIGENCE ANALYSIS")
    print("==============================================================")
    # Instantiate the agent controller pointing to our freshly updated lookup map
    swarm = FinancialAgentSwarm(lookup_path=LOOKUP_PATH)
    
    # Executes the token streaming output right into your terminal window console
    swarm.analyze_market_landscape(analysis_query, portfolio)


# =====================================================================
# SYSTEM MAIN CONTROL INTERFACE (Execution Orchestrator)
# =====================================================================

if __name__ == "__main__":
    # 1. Define the asset collection scope we want to target
    target_portfolio = ["TCS", "INFY"]
    
    # 2. Define the strategic question we want our multi-agent layer to evaluate
    user_prompt = "Compare the strategic positioning, margin trends, and AI automation investments mentioned by management."

    # 🚀 --- STEP 1: WEB TRANSCRIPT HARVESTING ---
    # Comment this out if you've already downloaded the files and want to save network bandwidth
    run_bulk_ingestion(target_portfolio)

    # 🚀 --- STEP 2: EXTRACT TEXT & GENERATE PARENT-CHILD CHUNKS ---
    # Comment out if you have already parsed the files and generated the data structures
    parents, children = run_parsing_and_hierarchical_chunking(target_portfolio)

    # 🚀 --- STEP 3: RE-COMPUTE EMBEDDINGS & WATER THE DATABASE ---
    # Comment out if your Chroma DB folder index is already completely up to date
    run_vector_store_hydration(children)

    # 🚀 --- STEP 4: TRIGGER LOCAL OLLAMA TWIN-AGENT REASONING SWARM ---
    # Keep this active to run queries against your financial intelligence index!
    execute_multi_agent_analysis(target_portfolio, user_prompt)
