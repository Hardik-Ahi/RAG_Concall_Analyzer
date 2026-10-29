import pymupdf
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Ingestion parsing engine initialized.")

    def extract_clean_text(self, pdf_path: str, start_at: int = 0) -> list[dict]:
        """
        Reads a local financial PDF page-by-page.
        Returns a list of dictionaries tracking page indexes and cleaned text segments.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []

        # Open document connection using PyMuPDF context manager
        with pymupdf.open(pdf_path) as doc:
            print(f"📖 Parsing {path_obj.name} ({len(doc)} pages total)...")
            
            for page_num in range(start_at, len(doc)):
                page = doc.load_page(page_num)

                # Define a new bounding box to ignore headers and footers
                # format: pymupdf.Rect(x0, y0, x1, y1) where (x0, y0) is top-left and (x1, y1) is bottom-right
                width_sub = 10
                height_sub = 50
                new_box = pymupdf.Rect(width_sub, height_sub, page.rect.width - width_sub, page.rect.height - height_sub)
                page.set_cropbox(new_box)
                
                # Extract text using structural layout mode to keep formatting tight
                text_content = page.get_text("text")
                
                # Basic cleaning: remove empty pages or short spacer markers
                if len(text_content.strip()) > 20:
                    parsed_pages.append({
                        "page_number": start_at + page_num + 1,  # real page no.
                        "content": text_content.strip(),
                        "source_file": path_obj.name
                    })
                    
        print(f"✅ Extracted data from {len(parsed_pages)} text-active pages.")
        return parsed_pages