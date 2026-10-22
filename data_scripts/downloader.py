import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import urllib3
from pypdf import PdfReader
import re

# Mute local network SSL warning logs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IndiaTranscriptDownloader:
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

    def fetch_latest_transcript(self, nse_symbol: str) -> bool:
        target_url = f"https://screener.in/company/{nse_symbol}/"
        print(f"📡 Scanning Screener for {nse_symbol}...")
        
        try:
            response = requests.get(target_url, headers=self.headers, verify=False, timeout=15)
            if response.status_code != 200:
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            documents_section = soup.find('section', {'id': 'documents'})
            if not documents_section: return False
                
            links = documents_section.find_all('a')
            transcript_url = None
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()
                if ("transcript" in text or "concall" in text) and href.endswith('.pdf'):
                    transcript_url = href
                    break
                    
            if not transcript_url:
                print(f"❌ No recent transcript PDFs found for {nse_symbol}")
                return False
                
            company_dir = self.download_dir / nse_symbol
            company_dir.mkdir(exist_ok=True)
            
            # Temporary file footprint name
            temp_file = company_dir / f"{nse_symbol}_temp.pdf"
            
            print(f"📥 Downloading binary stream...")
            pdf_bytes = requests.get(transcript_url, headers=self.headers, verify=False, stream=True, timeout=60)
            
            if pdf_bytes.status_code == 200:
                with open(temp_file, "wb") as f:
                    for chunk in pdf_bytes.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                # --- NEW METADATA TRACKING ENGINE IN ACTION ---
                year, month = self._extract_creation_date(temp_file)
                
                # Construct clean deterministic date filename
                final_filename = f"{nse_symbol}_Concall_{year}_{month}.pdf"
                final_path = company_dir / final_filename
                
                # Rename the file on your hard drive
                temp_file.rename(final_path)
                print(f"✅ Download complete & renamed dynamically to: {final_path.name}\n")
                return True
                
        except Exception as e:
            print(f"❌ Extraction infrastructure error: {str(e)}")
        return False

if __name__ == "__main__":
    downloader = IndiaTranscriptDownloader()
    target_stocks = ["TCS", "INFY", "RELIANCE"]
    for stock in target_stocks:
        downloader.fetch_latest_transcript(stock)
