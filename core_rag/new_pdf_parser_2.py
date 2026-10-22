import pdfplumber
from pathlib import Path

class FinancialPDFParser:
    def __init__(self):
        print("📁 Ingestion engine initialized with Multi-Line Row Repair.")

    def _convert_table_to_markdown(self, table_grid) -> str:
        """
        Converts a text grid into Markdown. 
        Detects broken rows caused by multi-line word wrapping and stitches them back together.
        """
        if not table_grid or len(table_grid) == 0:
            return ""

        # Step 1: Clean and standardize cells to strings
        normalized_grid = []
        for row in table_grid:
            if row:
                normalized_row = [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
                normalized_grid.append(normalized_row)

        if not normalized_grid:
            return ""

        num_cols = max(len(row) for row in normalized_grid)
        
        # Step 2: Algorithmic Re-Assembly of Wrapped Multi-line Cells (Strategy 1)
        reassembled_grid = []
        
        for row in normalized_grid:
            # Force rows to match maximum column width bounds safely
            while len(row) < num_cols:
                row.append("")
                
            # If Column 0 is empty but other columns contain values, it's a fractured wrap row!
            if reassembled_grid and row[0] == "" and any(cell != "" for cell in row[1:]):
                prev_row = reassembled_grid[-1]
                # Append the wrapped text segment directly to the cell content above it
                for idx in range(num_cols):
                    if row[idx]:
                        # Combine with a space to preserve readability
                        prev_row[idx] = f"{prev_row[idx]} {row[idx]}".strip()
            else:
                reassembled_grid.append(row)

        # Step 3: Construct the Final Markdown Grid string
        md_lines = []
        for r_idx, row in enumerate(reassembled_grid):
            # Final check to filter out completely blank stitched rows
            if not any(cell != "" for cell in row):
                continue
                
            sanitized_cells = [cell.replace("|", "\\|") for cell in row]
            md_lines.append("| " + " | ".join(sanitized_cells) + " |")

            if r_idx == 0:
                md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        return "\n" + "\n".join(md_lines) + "\n\n"

    def extract_layout_aware_text(self, pdf_path: str) -> list[dict]:
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found at target: {pdf_path}")
            return []

        parsed_pages = []

        # Configure pdfplumber geometric tolerances to prevent row splits (Strategy 2)
        custom_table_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_y_tolerance": 5,   # Merges horizontal baselines that are within 5 points
            "intersection_tolerance": 5
        }

        with pdfplumber.open(pdf_path) as pdf:
            print(f"📖 Re-parsing {path_obj.name} using Multi-Line Row Repair Engine...")
            
            for page_num, page in enumerate(pdf.pages[0:10]):  # Process first 10 pages for demonstration
                # Pass our precision layout override mapping
                extracted_tables = page.extract_tables(table_settings=custom_table_settings)
                
                page_text = page.extract_text(layout=False)
                if not page_text:
                    page_text = ""

                page_output = page_text

                if extracted_tables:
                    page_output += "\n\n### Extracted Financial Tables (Repaired Layout Flow)\n"
                    for table in extracted_tables:
                        markdown_table = self._convert_table_to_markdown(table)
                        page_output += markdown_table

                if len(page_output.strip()) > 20:
                    parsed_pages.append({
                        "page_number": page_num + 1,
                        "content": page_output.strip(),
                        "source_file": path_obj.name
                    })

        print(f"✅ Structural layout repair complete for {len(parsed_pages)} pages.")
        return parsed_pages

if __name__ == "__main__":
    parser = FinancialPDFParser()
    test_pdf = "financial_data/INFY/INFY_Annual_Report_2026.pdf"
    page_data = parser.extract_layout_aware_text(test_pdf)
    
    for page in page_data:
        if "### Extracted Financial Tables" in page["content"]:
            print(f"\n--- SUCCESS: Verifying Repaired Layout Row Alignment on Page {page['page_number']} ---")
            print(page["content"])
            break
