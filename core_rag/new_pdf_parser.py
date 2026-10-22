import pdfplumber
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Ingestion parsing engine migrated to pdfplumber backend.")

    def _convert_table_to_markdown(self, table_grid) -> str:
        """
        Converts a pdfplumber extracted 2D text matrix into a valid Markdown table.
        Preserves empty cell segments cleanly without breaking column alignments.
        """
        if not table_grid or len(table_grid) == 0:
            return ""

        # Filter out completely empty structural rows
        cleaned_grid = []
        for row in table_grid:
            if row and any(cell is not None and str(cell).strip() != "" for cell in row):
                cleaned_grid.append(row)

        if not cleaned_grid:
            return ""

        num_cols = max(len(row) for row in cleaned_grid)
        md_lines = []

        for r_idx, row in enumerate(cleaned_grid):
            row_cells = []
            for c_idx in range(num_cols):
                # Handle irregular row lengths or empty layout slots securely
                if c_idx < len(row) and row[c_idx] is not None:
                    # Clean up random inline newlines inside a single cell text block
                    cell_str = " ".join(str(row[c_idx]).split())
                else:
                    cell_str = ""

                # Sanitize raw markdown pipes to keep grid rendering functional
                cell_str = cell_str.replace("|", "\\|")
                row_cells.append(cell_str)

            md_lines.append("| " + " | ".join(row_cells) + " |")

            # Append the mandatory markdown alignment rules row right below the header
            if r_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n" + "\n".join(md_lines) + "\n\n"

    def extract_layout_aware_text(self, pdf_path: str) -> list[dict]:
        """
        Reads a financial PDF page-by-page, dynamically isolating corporate text
        and processing complex multi-column layout tables securely via pdfplumber.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []

        # Open file connection stream
        with pdfplumber.open(pdf_path) as pdf:
            print(f"📖 Parsing {path_obj.name} with pdfplumber layout layers ({len(pdf.pages)} pages)...")
            
            for page_num, page in enumerate(pdf.pages[33:35]):  # Process first 10 pages for demonstration
                # 1. Extract structural grids
                extracted_tables = page.extract_tables()
                
                # 2. Extract standard running layout text
                page_text = page.extract_text(layout=False)
                if not page_text:
                    page_text = ""

                page_output = page_text

                # 3. If structural grids are detected, format and append them cleanly
                if extracted_tables:
                    page_output += "\n\n### Extracted Financial Tables ###\n"
                    for table in extracted_tables:
                        markdown_table = self._convert_table_to_markdown(table)
                        page_output += markdown_table

                if len(page_output.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "content": page_output.strip(),
                        "source_file": path_obj.name
                    })

        print(f"✅ Structural layout parsing complete for {len(parsed_pages)} pages.")
        return parsed_pages

if __name__ == "__main__":
    parser = FinancialPDFParser()
    # Replace this string with your valid local file path
    test_pdf = "financial_data/INFY/INFY_Annual_Report_2026.pdf"
    
    page_data = parser.extract_layout_aware_text(test_pdf)
    print(page_data)
    
    for page in page_data:
        if "### Extracted Financial Tables ###" in page["content"]:
            print(f"\n--- SUCCESS: Verifying Page {page['page_number']} Layout Format ---")
            print(page["content"])  # Read out the formatted layout grid
            break
