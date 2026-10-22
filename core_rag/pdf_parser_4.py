import pymupdf
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Upgraded Ingestion parsing engine initialized with Grid-Mapping.")

    def _extract_clean_cell(self, page, bbox) -> str:
        """
        Extracts text from a single cell bounding box using a strict clipping mask.
        Keeps spacing, commas, decimals, and structural syntax strictly intact.
        """
        if not bbox or None in bbox or not (bbox is not None and hasattr(bbox, "__len__") and len(bbox) == 4):
            return ""
            
        # Extract text strictly from inside the grid bounds
        cell_text = page.get_text("text", clip=bbox)
        
        # Standardize structural gaps without stripping single spaces between words
        cleaned = " ".join(cell_text.split())
        return cleaned

    def _convert_table_to_markdown(self, page, table_object) -> str:
        """
        Maps a text grid to structural coordinate bounds to construct a stable Markdown output.
        Preserves empty cell segments cleanly.
        """
        # 1. Grab the clean layout structure (list of lists of strings)
        raw_text_grid = table_object.extract()
        if not raw_text_grid:
            return ""

        num_rows = len(raw_text_grid)
        num_cols = len(raw_text_grid[0]) if num_rows > 0 else 0
        
        if num_rows == 0 or num_cols == 0:
            return ""

        # 2. PyMuPDF stores all cell bboxes in a flat list of tuples matching the table layout.
        # Let's map that flat list into a matching 2D matrix [row][col] to align with our text grid.
        flat_cells = table_object.cells
        cell_bbox_matrix = []
        
        cell_idx = 0
        for r in range(num_rows):
            row_boxes = []
            for c in range(num_cols):
                if cell_idx < len(flat_cells):
                    row_boxes.append(flat_cells[cell_idx])
                    cell_idx += 1
                else:
                    #print("appending NONE")
                    row_boxes.append(None)
            cell_bbox_matrix.append(row_boxes)

        # 3. Iterate through our text grid and pull clean text using cell coordinate clipping masks
        md_lines = []
        
        for r_idx, row_text in enumerate(raw_text_grid):
            row_cells = []
            for c_idx, cell_string in enumerate(row_text):
                index = r_idx * num_cols + c_idx
                bbox = cell_bbox_matrix[index] if index < len(cell_bbox_matrix) and cell_bbox_matrix[index] is not None else None
                
                # Check if it's a visually empty cell placeholder
                if cell_string is None or str(cell_string).strip() == "":
                    row_cells.append("")
                elif bbox:
                    # Precision fetch using the coordinate layout bounding box
                    clean_string = self._extract_clean_cell(page, bbox)
                    # If clipping returns empty strings, fall back to the text grid data safely
                    row_cells.append(clean_string if clean_string else str(cell_string).strip())
                else:
                    row_cells.append(str(cell_string).strip())
            
            # Sanitize raw markdown divider characters to prevent alignment breaks
            sanitized_cells = [c.replace("|", "\\|") for c in row_cells]
            md_lines.append("| " + " | ".join(sanitized_cells) + " |")
            
            # Inject standard markdown alignment rules directly below the header row
            if r_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n" + "\n".join(md_lines) + "\n\n"

    def extract_layout_aware_text(self, pdf_path: str) -> list[dict]:
        """
        Reads a financial PDF page-by-page, dynamically isolating corporate text
        and processing multi-column layout tables flawlessly.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []
        
        with pymupdf.open(pdf_path) as doc:
            print(f"📖 Re-parsing {path_obj.name} with grid mapping layers ({len(doc)} pages)...")
            
            for page_num in range(len(doc) // 10):
                page = doc.load_page(page_num)
                
                # Automatically extract structural table models
                tables = page.find_tables()
                page_output = ""
                
                if tables and len(tables.tables) > 0:
                    raw_text = page.get_text("text")
                    page_output += raw_text + "\n\n### Extracted Financial Tables ###\n"
                    
                    for tab in tables.tables:
                        markdown_table = self._convert_table_to_markdown(page, tab)
                        page_output += markdown_table
                else:
                    page_output = page.get_text("text")

                if len(page_output.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "content": page_output.strip(),
                        "source_file": path_obj.name
                    })
                    
        print(f"✅ Structural grid parsing complete for {len(parsed_pages)} pages.")
        return parsed_pages

if __name__ == "__main__":
    parser = FinancialPDFParser()
    # Replace this string with a valid test path pointing to your downloaded reports
    test_pdf = "financial_data/INFY/INFY_Annual_Report_2026.pdf"
    
    page_data = parser.extract_layout_aware_text(test_pdf)
    output_path = "page_data_3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved parsed page data to {output_path}")
    
    for page in page_data:
        if "### Extracted Financial Tables" in page["content"]:
            print(f"\n--- SUCCESS: Verifying Page {page['page_number']} Layout Syntax ---")
            print(page["content"])  # Read out the formatted layout grid
            break
