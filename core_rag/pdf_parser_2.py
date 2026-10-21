import json
import pymupdf
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Ingestion parsing engine initialized with Table Preservation.")

    def _convert_table_to_markdown(self, table_data) -> str:
        """
        Converts a raw 2D array of rows and cells into a clean Markdown table string.
        """
        if not table_data or not table_data[0]:
            return ""
            
        rows = table_data
        headers = [str(cell).strip() if cell else "" for cell in rows[0]]
        
        # Build the Markdown header row
        md = "\n| " + " | ".join(headers) + " |\n"
        # Build the alignment/divider row
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Build out the financial data rows
        for row in rows[1:]:
            cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
            md += "| " + " | ".join(cleaned_row) + " |\n"
            
        return md + "\n"

    def extract_layout_aware_text(self, pdf_path: str) -> list[dict]:
        """
        Reads a local financial PDF page-by-page, explicitly detecting 
        and formatting tables into Markdown grids.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []
        
        with pymupdf.open(pdf_path) as doc:
            print(f"📖 Restructuring {path_obj.name} ({len(doc)} pages)...")
            
            for page_num in range(len(doc) // 10):
                page = doc.load_page(page_num)
                
                # 1. Attempt to locate visual tabular boundaries
                tables = page.find_tables()
                
                # Initialize text container
                page_output = ""
                
                if tables and len(tables.tables) > 0:
                    # Extract standard text excluding the bounding boxes of tables to prevent duplicates
                    # For simplicity in PyMuPDF, we can grab the text and append structural tables cleanly
                    raw_text = page.get_text("text")
                    page_output += raw_text + "\n\n### Extracted Tables ###\n"
                    
                    # Convert each found table structural array to markdown text block
                    for tab in tables.tables:
                        raw_table_array = tab.extract()
                        markdown_table = self._convert_table_to_markdown(raw_table_array)
                        page_output += markdown_table
                else:
                    # No tables found, process as plain text page layout
                    page_output = page.get_text("text")

                if len(page_output.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "content": page_output.strip(),
                        "source_file": path_obj.name
                    })
                    
        print(f"✅ Structural parsing complete for {len(parsed_pages)} pages.")
        return parsed_pages

if __name__ == "__main__":
    parser = FinancialPDFParser()
    test_pdf = "financial_data/INFY/INFY_Annual_Report_2026.pdf"
    
    page_data = parser.extract_layout_aware_text(test_pdf)
    output_path = "page_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved parsed page data to {output_path}")
    
    # Scan through pages until we find one that had a table to print out as validation
    for page in page_data:
        if "### Extracted Tables ###" in page["content"]:
            print(f"\n--- SUCCESS: Table Preserved on Page {page['page_number']} ---")
            print(page["content"]) # Look at the top portion of the text and markdown conversion
            break
