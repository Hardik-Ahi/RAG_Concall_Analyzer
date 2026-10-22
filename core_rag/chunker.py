import re
import uuid
from pathlib import Path

class HierarchicalTranscriptChunker:
    def __init__(self):
        # Looks for capitalized words or titles sitting on a standalone line (speaker names)
        self.speaker_pattern = re.compile(r"^[A-Z][a-zA-Z']+\s[A-Z][a-zA-Z']+$")

    def _extract_meta_from_filename(self, filename: str) -> dict:
        """
        Parses our clean filename convention (e.g., TCS_Concall_2025_07.pdf)
        to extract baseline metadata attributes.
        """
        meta = {"ticker": "UNKNOWN", "year": "UNKNOWN", "month": "UNKNOWN"}
        # Regex to pull ticker, year, and month out of the filename format
        match = re.search(r'([A-Z0-9]+)_Concall_(\d{4}|\d{2}|UNKNOWN)_(\d{2}|UNKNOWN)', filename)
        if match:
            meta["ticker"] = match.group(1)
            meta["year"] = match.group(2)
            meta["month"] = match.group(3)
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
                "speaker": block["speaker"],
                "parent_text": f"{block['speaker']}: {block['text']}"
            })

        print(f"🧩 Successfully synthesized {len(parent_chunks)} structural Parent Chunks.")
        return parent_chunks

if __name__ == "__main__":
    # Test execution using our previous FinancialPDFParser module
    from pdf_parser import FinancialPDFParser
    
    # Target one of our safely downloaded local files
    parser = FinancialPDFParser()
    test_pdf_path = "financial_data/INFY/INFY_Concall_2026_07.pdf" # Replace with your actual local file
    
    if Path(test_pdf_path).exists():
        pages = parser.extract_clean_text(test_pdf_path, start_at=2)  # skip first 2 pages of INFY concall
        chunker = HierarchicalTranscriptChunker()
        
        parents = chunker.build_parent_chunks(pages)
        if parents:
            index = 1
            print("\n--- SAMPLE PARENT CHUNK (UUID Tracking Proof) ---")
            print(f"Parent ID: {parents[index]['parent_id']}")
            print(f"Metadata: Ticker={parents[index]['ticker']} | Date={parents[index]['year']}-{parents[index]['month']}")
            print(f"Speaker: {parents[index]['speaker']}")
            print(f"Snippet: {parents[index]['parent_text']}")
