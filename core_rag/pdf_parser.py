import pymupdf
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Ingestion parsing engine initialized.")

    def extract_clean_text(self, pdf_path: str) -> list[dict]:
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
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text using structural layout mode to keep formatting tight
                text_content = page.get_text("text")
                
                # Basic cleaning: remove empty pages or short spacer markers
                if len(text_content.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "raw_text": text_content.strip(),
                        "source_file": path_obj.name
                    })
                    
        print(f"✅ Extracted data from {len(parsed_pages)} text-active pages.")
        return parsed_pages

if __name__ == "__main__":
    # Test execution against one of your downloaded Indian stock PDFs
    parser = FinancialPDFParser()
    
    # Path to one of the PDFs downloaded by your script in Approach B
    test_pdf = "financial_data/INFY/INFY_Concall_2026_07.pdf"
    
    # Let's inspect the first couple of pages extracted
    page_data = parser.extract_clean_text(test_pdf)
    if page_data:
        print("\n--- SAMPLE EXTRACT (Page 1) ---")
        print(page_data[5]["raw_text"])
        print("-------------------------------")
