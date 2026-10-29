import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import urllib3
from pypdf import PdfReader
import re

# Mute local network SSL warning logs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ConcallDownloader:
    def __init__(self, download_dir: str = "financial_data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def _extract_creation_date(self, pdf_path: Path) -> tuple[str, str]:
        """
        Reads the local PDF's metadata properties to extract the Creation Year and Month.
        Falls back to 'UNKNOWN' if headers are stripped.
        """
        try:
            reader = PdfReader(pdf_path)
            meta = reader.metadata
            
            # PDF Creation dates are standard strings formatted like: 'D:20250718143000'
            creation_date_str = meta.get('/CreationDate', '')
            
            if creation_date_str and len(creation_date_str) >= 8:
                # Use regex to isolate the first 6 digits (YYYYMM) after the 'D:' prefix
                match = re.search(r'D?:?(\d{4})(\d{2})', creation_date_str)
                if match:
                    year, month = match.group(1), match.group(2)
                    return year, month
                    
        except Exception as e:
            print(f"⚠️ Could not read internal metadata metadata boundaries: {str(e)}")
            
        return "UNKNOWN", "UNKNOWN"

    def fetch_all_transcripts(self, nse_symbol: str) -> bool:
        """
        Dynamically extracts and processes ALL listed Earnings Call Transcripts 
        available on Screener.in for a target company in a single run.
        """
        target_url = f"https://screener.in/company/{nse_symbol}/"
        print(f"📡 Scanning Screener for {nse_symbol} documents collection...")
        
        try:
            response = requests.get(target_url, headers=self.headers, verify=False, timeout=15)
            if response.status_code != 200:
                print(f"❌ Aborted: Could not connect to page for {nse_symbol} (Status: {response.status_code})")
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the documents section wrapper container
            documents_section = soup.find('section', {'id': 'documents'})
            if not documents_section: 
                print(f"❌ Documents block section layout not found for {nse_symbol}")
                return False
                
            links = documents_section.find_all('a')
            valid_transcript_urls = []
            
            # Step 1: Scan the page layout and gather all available transcript links
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()
                
                if ("transcript" in text or "concall" in text) and href.endswith('.pdf'):
                    # Append unique URLs to our batch execution queue array list
                    if href not in valid_transcript_urls:
                        valid_transcript_urls.append(href)
            
            if not valid_transcript_urls:
                print(f"❌ No historical transcript PDFs found listed for {nse_symbol}")
                return False
                
            print(f"📋 Found {len(valid_transcript_urls)} transcripts available for {nse_symbol}. Initializing batch stream processing...")
            
            company_dir = self.download_dir / nse_symbol
            company_dir.mkdir(exist_ok=True)
            
            download_count = 0
            
            # Step 2: Loop through every found document and execute binary streaming
            for idx, transcript_url in enumerate(valid_transcript_urls):
                print(f"📥 Processing [{idx+1}/{len(valid_transcript_urls)}]: Streaming binary packet stream...")
                
                try:
                    # Create unique temporary footprint file name for this sequence run
                    temp_file = company_dir / f"{nse_symbol}_temp_{idx}.pdf"
                    
                    pdf_bytes = requests.get(transcript_url, headers=self.headers, verify=False, stream=True, timeout=60)
                    
                    if pdf_bytes.status_code == 200:
                        with open(temp_file, "wb") as f:
                            for chunk in pdf_bytes.iter_content(chunk_size=8192):
                                if chunk: f.write(chunk)
                        
                        # --- EXECUTING YOUR METADATA DATE DISCOVERY PIPELINE ---
                        year, month = self._extract_creation_date(temp_file)
                        
                        # Construct a clean, unique name incorporating timestamps and safety indexes
                        final_filename = f"{nse_symbol}_Concall_{year}_{month}_{idx+1}.pdf"
                        final_path = company_dir / final_filename
                        
                        # Prevent collision errors if a file with that name already exists on your hard drive
                        if final_path.exists():
                            temp_file.unlink() # Delete the redundant temp download footprint
                            print(f"⏭️ File already archived locally: {final_filename}")
                            download_count += 1
                            continue
                            
                        temp_file.rename(final_path)
                        print(f"✅ Success: Downloaded & organized to -> {final_filename}")
                        download_count += 1
                    else:
                        print(f"⚠️ Warning: Endpoint rejected document request (Status: {pdf_bytes.status_code})")
                        if temp_file.exists(): temp_file.unlink()
                        
                except Exception as stream_err:
                    print(f"⚠️ Failed to complete stream download entry index {idx+1}: {str(stream_err)}")
                    
            print(f"📊 Batch execution sync sequence closed. Archived {download_count}/{len(valid_transcript_urls)} files for {nse_symbol}.\n")
            return download_count > 0
                
        except Exception as e:
            print(f"❌ Extraction infrastructure pipeline error: {str(e)}")
        return False
