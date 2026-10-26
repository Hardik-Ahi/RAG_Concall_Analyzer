import re
import uuid
from pathlib import Path
import json
import os
from core_rag.vector_store import LocalVectorStorage

class HierarchicalTranscriptChunker:
    def __init__(self):
        # Looks for FirstName LastName sitting on a standalone line (speaker names)
        self.speaker_pattern = re.compile(r"^[A-Z][a-zA-Z']+\s[A-Z][a-zA-Z']+$")

    def _extract_meta_from_filename(self, filename: str) -> dict:
        """
        Parses our clean filename convention (e.g., TCS_Concall_2025_07.pdf)
        to extract baseline metadata attributes.
        """
        meta = {"ticker": "UNKNOWN", "year": "UNKNOWN", "month": "UNKNOWN", "index": "UNKNOWN"}
        # Regex to pull ticker, year, and month out of the filename format
        match = re.search(r'([A-Z0-9]+)_Concall_(\d{4}|\d{2}|UNKNOWN)_(\d{2}|UNKNOWN)_(\d+|UNKNOWN)', filename)
        if match:
            meta["ticker"] = match.group(1)
            meta["year"] = match.group(2)
            meta["month"] = match.group(3)
            meta["index"] = match.group(4)  # there might be >1 concalls in one month
        return meta

    def build_parent_chunks(self, parsed_pages: list[dict]) -> list[dict]:
        """
        Converts raw text pages into structurally bounded Parent Chunks 
        by grouping consecutive text blocks belonging to a single speaker.
        """
        if not parsed_pages:
            return []

        filename = parsed_pages[0].get("source_file", "")
        file_meta = self._extract_meta_from_filename(filename)
        
        raw_speech_blocks = []
        current_speaker = "MANAGEMENT/MODERATOR"
        current_block_text = []

        # Step 1: Chronologically combine text flows across pages
        for page in parsed_pages:
            lines = page["content"].split("\n")
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                # Check if this line signals a new speaker change
                speaker_match = self.speaker_pattern.match(line_str)
                if speaker_match:
                    # Save the previous speaker's compiled block before switching
                    if current_block_text:
                        raw_speech_blocks.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_block_text)
                        })
                    
                    current_speaker = speaker_match.group(0).strip()
                    # Strip the speaker name off the text segment
                    remaining_text = line_str[speaker_match.end():].strip()
                    current_block_text = [remaining_text] if remaining_text else []
                else:
                    current_block_text.append(line_str)

        # Append the final trailing block
        if current_block_text:
            raw_speech_blocks.append({
                "speaker": current_speaker,
                "text": " ".join(current_block_text)
            })

        # Step 2: Structure speech blocks into complete unique Parent Chunks
        parent_chunks = []
        for block in raw_speech_blocks:
            if len(block["text"].strip()) < 30: # Skip empty spacer text blocks
                continue
                
            parent_id = str(uuid.uuid4())
            parent_chunks.append({
                "parent_id": parent_id,
                "ticker": file_meta["ticker"],
                "year": file_meta["year"],
                "month": file_meta["month"],
                "index": file_meta["index"],
                "speaker": block["speaker"],
                "parent_text": f"{block['speaker']}: {block['text']}"
            })

        print(f"🧩 Successfully synthesized {len(parent_chunks)} structural Parent Chunks.")
        return parent_chunks

    def build_child_chunks(self, parent_chunks: list[dict], chunk_size: int = 150, overlap: int = 40) -> list[dict]:
        """
        Slices each parent text into overlapping small child segments.
        Prepends a strict Context Enrichment Header to ensure accurate vector matching.
        """
        child_chunks = []

        for parent in parent_chunks:
            # Tokenise/split text into words
            words = parent["parent_text"].split()

            # Sliding window execution logic loop (works also on len(words) < chunk_size [slicing handles it])
            start_idx = 0
            while start_idx < len(words):
                end_idx = start_idx + chunk_size
                window_words = words[start_idx:end_idx]
                
                header = f"Context: {parent['ticker']} Concall ({parent['year']}-{parent['month']}) | Speaker: {parent['speaker']}\n---\n"
                enriched_text = header + " ".join(window_words)
                
                child_chunks.append({
                    "child_id": str(uuid.uuid4()),
                    "parent_id": parent["parent_id"],
                    "ticker": parent["ticker"],
                    "year": parent["year"],
                    "month": parent["month"],
                    "index": parent["index"],
                    "speaker": parent["speaker"],
                    "enriched_text": enriched_text
                })
                
                # Move window forward by chunk_size minus the overlap window
                start_idx += (chunk_size - overlap)
                
                # Prevent trailing tiny scraps if we reach the end of the text array bounds
                if start_idx + overlap >= len(words):
                    break

        print(f"👶 Generated {len(child_chunks)} Enriched Child Chunks across parents.")
        return child_chunks

    def save_parent_lookup(self, parent_chunks: list[dict], storage_path: str = "parent_lookup.json"):
        # Convert list to a fast key-value dictionary mapping parent_id -> parent_text
        lookup_dict = {p["parent_id"]: p["parent_text"] for p in parent_chunks}
        
        # Load existing lookups if they exist so we don't overwrite older runs
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    existing = json.load(f)
                    lookup_dict.update(existing)
            except Exception:
                pass
                
        with open(storage_path, "w") as f:
            json.dump(lookup_dict, f, indent=4)
        print(f"💾 Relational lookup dictionary saved to '{storage_path}'")


if __name__ == "__main__":
    # Internal validation test loop to scale chunking across all downloaded files
    from pdf_parser import FinancialPDFParser
    
    parser = FinancialPDFParser()
    chunker = HierarchicalTranscriptChunker()
    vector_db = LocalVectorStorage()
    
    # Track the base path containing all our company directories
    transcript_root = Path("financial_data")
    
    # Find all files matching the pattern * _Concall_*.pdf recursively across all stock folders
    all_pdf_paths = list(transcript_root.glob("**/*_Concall_*.pdf"))
    
    if not all_pdf_paths:
        print(f"❌ No valid date-stamped transcript PDFs found inside the '{transcript_root}' directory tree.")
    else:
        print(f"📚 Found {len(all_pdf_paths)} corporate transcripts across directories. Starting batch hierarchical chunking processing...")
        
        global_parents = []
        global_children = []
        
        for idx, pdf_path in enumerate(all_pdf_paths):
            print(f"\n────────────────────────────────────────────────────────────")
            print(f"🔄 Processing File [{idx+1}/{len(all_pdf_paths)}]: {pdf_path.name}")
            print(f"────────────────────────────────────────────────────────────")
            
            try:
                # Extract layout-aware plain text page arrays chronologically
                pages = parser.extract_clean_text(str(pdf_path))
                if not pages:
                    print(f"⚠️ Warning: Page extraction empty or failed for {pdf_path.name}. Skipping file.")
                    continue
                
                # Segment into independent Parent chunks by tracking speaker turn signals
                file_parents = chunker.build_parent_chunks(pages)
                
                # Construct sub-segment Child chunks with overlapping text window matrices
                file_children = chunker.build_child_chunks(file_parents)
                
                # Aggregate findings into our master batch execution tracking arrays
                global_parents.extend(file_parents)
                global_children.extend(file_children)
                
            except Exception as file_err:
                print(f"❌ Failed to process data structure layers for {pdf_path.name}: {str(file_err)}")
        
        print(f"\n📊 Batch Processing Complete!")
        print(f"├── Total Unique Parent Context Blocks Synthesized: {len(global_parents)}")
        print(f"└── Total Enriched Child Tokens Generated: {len(global_children)}")
        
        # Save or incrementally update our local lookup map on your hard drive
        if global_parents:
            chunker.save_parent_lookup(global_parents, storage_path="parent_lookup.json")
            print(f"💾 Centralized relational lookup map updated cleanly at 'parent_lookup.json'")
            vector_db.index_child_chunks(global_children)
            print(f'🚀 Vector database hydrated with {len(global_children)} child segments for semantic search.')
            '''
            try:
            except AttributeError:
                # Fallback if save_parent_lookup is standalone inside main or chunker spaces
                lookup_path = "parent_lookup.json"
                lookup_dict = {p["parent_id"]: p["parent_text"] for p in global_parents}
                if os.path.exists(lookup_path):
                    try:
                        with open(lookup_path, "r") as f:
                            lookup_dict.update(json.load(f))
                    except Exception: pass
                with open(lookup_path, "w") as f:
                    json.dump(lookup_dict, f, indent=4)
            '''


'''
if __name__ == "__main__":
    from pdf_parser import FinancialPDFParser
    
    parser = FinancialPDFParser()
    test_pdf_path = "financial_data/INFY/INFY_Concall_2026_07.pdf" # Replace with your actual local file
    
    if Path(test_pdf_path).exists():
        pages = parser.extract_clean_text(test_pdf_path, start_at=2)  # skip first 2 pages of INFY concall
        chunker = HierarchicalTranscriptChunker()
        
        parents = chunker.build_parent_chunks(pages)
        children = chunker.build_child_chunks(parents)
        chunker.save_parent_lookup(parents, storage_path="parent_lookup.json")

        if parents:
            parents_index = 1
            print("\n--- SAMPLE PARENT CHUNK (UUID Tracking Proof) ---")
            print(f"Parent ID: {parents[parents_index]['parent_id']}")
            print(f"Metadata: Ticker={parents[parents_index]['ticker']} | Date={parents[parents_index]['year']}-{parents[parents_index]['month']}")
            print(f"Speaker: {parents[parents_index]['speaker']}")
            print(f"Snippet: {parents[parents_index]['parent_text']}")
        
        if children:
            children_index = 3
            print("\n--- SAMPLE ENRICHED CHILD CHUNK PROOF ---")
            print(f"Child ID: {children[children_index]['child_id']}")
            print(f"Relational Parent ID Link: {children[children_index]['parent_id']}")
            print(f"Enriched Text Content:\n{children[children_index]['enriched_text']}")
            print("----------------------------------------")
'''