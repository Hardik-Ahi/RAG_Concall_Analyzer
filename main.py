import os
from pathlib import Path
from dotenv import load_dotenv
from core_rag.pdf_parser import FinancialPDFParser
from core_rag.chunker import HierarchicalTranscriptChunker
from core_rag.vector_store import LocalVectorStorage

load_dotenv()  # Load environment variables from .env file

def run_end_to_end_pipeline(target_tickers: list[str]):
    print("🎬 STARTING INDIAN FINANCIAL RAG INGESTION PIPELINE 🎬\n")
    
    # 1. Initialize all functional subsystems
    parser = FinancialPDFParser()
    chunker = HierarchicalTranscriptChunker()
    db = LocalVectorStorage()
    
    # Track paths of files ready to process
    downloaded_files = []
    
    # --- PHASE 1: LIST PDFs ---
    print("\n--- PHASE 1: FETCHING TRANSCRIPT DATA FROM THE WEB ---")
    for ticker in target_tickers:
        # Locate the newly downloaded file inside the company folder
        company_dir = Path("financial_data") / ticker
        # Grab the actual PDF file path (ignoring any temporary placeholders)
        pdf_paths = list(company_dir.glob("*_Concall_*.pdf"))
        if pdf_paths:
            downloaded_files.append(pdf_paths[0]) # Snag the latest one
                
    if not downloaded_files:
        print("❌ No files were successfully downloaded. Pipeline aborted.")
        return

    # --- PHASE 2 & 3: PARSING & HIERARCHICAL CHUNKING ---
    print("\n--- PHASE 2 & 3: STRUCTURAL LAYOUT PROCESSING & CHUNKING ---")
    all_compiled_parents = []
    all_compiled_children = []
    
    for pdf_path in downloaded_files:
        # Step A: Parse pages to text chunks
        parsed_pages = parser.extract_clean_text(str(pdf_path))
        if not parsed_pages:
            continue
            
        # Step B: Segment into Parent blocks by speaker
        parent_chunks = chunker.build_parent_chunks(parsed_pages)
        
        # Step C: Generate overlapping Child segments with Context Enrichment Headers
        child_chunks = chunker.build_child_chunks(parent_chunks)
        
        all_compiled_parents.extend(parent_chunks)
        all_compiled_children.extend(child_chunks)

    # Save the parent lookup dictionary
    chunker.save_parent_lookup(all_compiled_parents, storage_path="parent_lookup.json")

    # --- PHASE 4: LOCAL VECTOR STORE SYNC ---
    print("\n--- PHASE 4: EMBEDDING GENERATION & DATABASE SYNC ---")
    if all_compiled_children:
        db.index_child_chunks(all_compiled_children)
        print(f"\n🚀 SUCCESS! Database hydrated with {len(all_compiled_children)} vectors across {len(target_tickers)} companies.")
    else:
        print("❌ Data assembly yielded 0 valid child chunks for processing.")
        return

    # --- PHASE 5: VERIFICATION QUERY ---
    print("\n--- PHASE 5: RUNNING LIVE SEMANTIC SEARCH VERIFICATION ---")
    sample_query = "What did management say about AI strategy, margins, or automation?"
    print(f"User Query: '{sample_query}'")
    
    # Run a test vector lookup targeting one specific company from our index list
    test_ticker = target_tickers[0]
    search_results = db.query_database(query_text=sample_query, ticker=test_ticker, top_k=2)
    
    print(f"\n🎯 Top semantic matches found inside '{test_ticker}' index zone:")
    for idx, doc in enumerate(search_results["documents"][0]):
        meta = search_results["metadatas"][0][idx]
        print(f"\nMatch #{idx+1} [Speaker: {meta['speaker']} | Year: {meta['year']}]:")
        print(f"{doc}")
        print("-" * 40)
    
def query_database(query_text: str, ticker: str = None, year: str = None, top_k: int = 3) -> dict:
    """
    Queries the database using semantic search while applying optional hard metadata filters.
    """
    db = LocalVectorStorage()
    
    # Run a test vector lookup targeting one specific company from our index list
    test_ticker = 'INFY'
    search_results = db.query_database(query_text=query_text, ticker=test_ticker, top_k=5)
    
    print(f"\n🎯 Top semantic matches found inside '{test_ticker}' index zone:")
    for idx, doc in enumerate(search_results["documents"][0]):
        meta = search_results["metadatas"][0][idx]
        print(f"\nMatch #{idx+1} [Speaker: {meta['speaker']} | Year: {meta['year']}]:")
        print(f"{doc}")
        print("-" * 40)

if __name__ == "__main__":
    # Define the Indian equities you want to index for analysis
    portfolio = ["INFY"]
    query_database("How did the company's AI strategy evolve?")
