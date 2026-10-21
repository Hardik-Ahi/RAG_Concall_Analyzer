import pymupdf
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Upgraded Ingestion parsing engine initialized with Precision Cell Clipping.")

    def _extract_clean_cell(self, page, bbox) -> str:
        """
        Extracts text from a single cell bounding box using a strict clipping mask.
        This forces the layout engine to keep inline spaces, periods, and commas intact.
        """
        if not bbox:
            return ""
            
        # Extract text strictly from inside the cell coordinates
        cell_text = page.get_text("text", clip=bbox)
        
        # Clean up any chaotic whitespace/newlines within the single cell 
        # while explicitly preserving single spaces between words
        cleaned = " ".join(cell_text.split())
        return cleaned

    def _convert_table_to_markdown(self, page, table_object) -> str:
        """
        Uses precision coordinate clipping to format table objects into perfect Markdown.
        """
        # table_object.cells is a 1D or 2D structural array containing cell objects with coordinates
        grid = table_object.extract()
        if not grid:
            return ""

        num_rows = len(grid)
        num_cols = len(grid[0]) if num_rows > 0 else 0
        
        if num_rows == 0 or num_cols == 0:
            return ""

        # Initialize markdown generation
        md_lines = []
        
        # Step through rows and columns geometrically using structural cell components
        # table_object.rows gives us sequential row data structures containing cell models
        for r_idx, row in enumerate(table_object.rows):
            row_cells = []
            for c_idx, cell in enumerate(row.cells):
                if cell and cell.bbox:
                    # Precision extraction using the bounding box clip
                    cell_string = self._extract_clean_cell(page, cell.bbox)
                else:
                    cell_string = ""
                
                # Sanitize markdown delimiters out of cell contents to prevent broken tables
                cell_string = cell_string.replace("|", "\\|")
                row_cells.append(cell_string)
            
            # Construct the line block
            md_lines.append("| " + " | ".join(row_cells) + " |")
            
            # Append markdown alignment divider directly below the header row
            if r_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n" + "\n".join(md_lines) + "\n\n"

    def extract_layout_aware_text(self, pdf_path: str) -> list[dict]:
        """
        Reads an Indian financial PDF page-by-page, extracting structural text
        and preserving multi-column charts/tables flawlessly via coordinate clipping.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []
        
        with pymupdf.open(pdf_path) as doc:
            print(f"📖 Re-parsing {path_obj.name} with precision clipping layers ({len(doc)} pages)...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Detect structural tables visually
                tables = page.find_tables()
                page_output = ""
                
                if tables and len(tables.tables) > 0:
                    # Pull base page text
                    raw_text = page.get_text("text")
                    page_output += raw_text + "\n\n### Extracted Financial Tables (Preserved Structural Layout)\n"
                    
                    # Process each isolated grid layout sequentially using coordinate loops
                    for tab in tables.tables:
                        markdown_table = self._convert_table_to_markdown(page, tab)
                        page_output += markdown_table
                else:
                    # Clean native page text layout
                    page_output = page.get_text("text")

                if len(page_output.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "content": page_output.strip(),
                        "source_file": path_obj.name
                    })
                    
        print(f"✅ Structural parsing successfully completed for {len(parsed_pages)} pages.")
        return parsed_pages

if __name__ == "__main__":
    parser = FinancialPDFParser()
    test_pdf = "raw_data_india/INFY/INFY_Annual_Report.pdf"
    
    page_data = parser.extract_layout_aware_text(test_pdf)
    
    # Verify the output against the table sections
    for page in page_data:
        if "### Extracted Financial Tables" in page["content"]:
            print(f"\n--- VERIFICATION: Precision Text Check on Page {page['page_number']} ---")
            print(page["content"][-1000:])  # View the newly rendered clipped markdown text block
            break
