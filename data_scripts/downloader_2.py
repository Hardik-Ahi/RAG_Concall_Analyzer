import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import urllib3

# Mute local network SSL warning logs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IndiaTranscriptDownloader:
    def __init__(self, download_dir: str = "financial_data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Standard browser headers to ensure requests pass through corporate servers smoothly
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

    def fetch_latest_transcript(self, nse_symbol: str) -> bool:
        """
        Dynamically extracts and saves the latest open-source 
        Earnings Call Transcript PDF for any listed Indian ticker.
        """
        target_url = f"https://screener.in/company/{nse_symbol}/"
        print(f"📡 Scanning Screener for {nse_symbol} Earnings Disclosures...")
        
        try:
            response = requests.get(target_url, headers=self.headers, verify=False, timeout=15)
            if response.status_code != 200:
                print(f"❌ Aborted: Target stock page inaccessible (Status: {response.status_code})")
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            documents_section = soup.find('section', {'id': 'documents'})
            
            if not documents_section:
                print(f"❌ No document block found on page layout for {nse_symbol}")
                return False
                
            links = documents_section.find_all('a')
            transcript_url = None
            transcript_text = ""
            
            # Step through links sequentially to locate transcripts
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                normalized_text = text.lower()
                
                # We specifically look for 'concall transcript' tags rather than dry annual reports
                if ("transcript" in normalized_text or "concall" in normalized_text) and href.endswith('.pdf'):
                    transcript_url = href
                    transcript_text = text
                    break # Snag the absolute latest one at the top of the feed
                    
            if not transcript_url:
                print(f"❌ No recent transcript PDFs found indexed for {nse_symbol}")
                return False
                
            # Create isolated directory path for the stock
            company_dir = self.download_dir / nse_symbol
            company_dir.mkdir(exist_ok=True)
            year_match = re.search(r"\b(\d{4})\b", transcript_text or transcript_url)
            year_suffix = f"{year_match.group(1)}" if year_match else ""
            output_file = company_dir / f"{nse_symbol}_Transcript_{year_suffix}.pdf"
            
            print(f"📥 Found valid stream path. Downloading binary file...")
            pdf_bytes = requests.get(transcript_url, headers=self.headers, verify=False, stream=True, timeout=60)
            
            if pdf_bytes.status_code == 200:
                with open(output_file, "wb") as f:
                    for chunk in pdf_bytes.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                print(f"✅ Download complete: {output_file}\n")
                return True
                
        except Exception as e:
            print(f"❌ Infrastructure exception tracking {nse_symbol}: {str(e)}")
            
        return False

if __name__ == "__main__":
    downloader = IndiaTranscriptDownloader()
    
    # Try any Indian stock symbol to verify the loop
    target_stocks = ["TCS", "INFY", "RELIANCE"]
    
    for stock in target_stocks:
        downloader.fetch_latest_transcript(stock)
