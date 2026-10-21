import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional, Tuple
import urllib3

# Suppress SSL warnings in the terminal output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ScreenerPDFDownloader:
    def __init__(self, download_dir: str = "financial_data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Mirror a natural web browser layout to prevent being blocked by corporate firewalls
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }

    def _get_annual_report_details(self, nse_symbol: str) -> Optional[Tuple[str, str]]:
        """
        Scrapes Screener.in for a specific company symbol to extract the latest official Annual Report link.
        """
        screener_url = f"https://www.screener.in/company/{nse_symbol}/"
        print(f"🔍 Searching Screener.in for symbol: {nse_symbol}...")
        
        try:
            # verify=False bypasses local SSL chain verification blocks
            response = requests.get(screener_url, headers=self.headers, verify=False, timeout=15)
            if response.status_code != 200:
                print(f"❌ Could not access page for {nse_symbol} (Status: {response.status_code})")
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Target the document links section on Screener
            documents_section = soup.find('section', {'id': 'documents'})
            if not documents_section:
                print(f"❌ Documents block not found for {nse_symbol}")
                return None
                
            # Locate all hyperlink anchors pointing directly to PDF objects
            links = documents_section.find_all('a')
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Snag the first valid link matching an Annual Report
                if "annual report" in text.lower() and href.endswith('.pdf'):
                    return href, text
                    
            print(f"❌ No direct PDF annual report links found for {nse_symbol}")
            return None
            
        except Exception as e:
            print(f"❌ Error scraping metadata for {nse_symbol}: {str(e)}")
            return None

    def get_annual_report_url(self, nse_symbol: str) -> Optional[str]:
        details = self._get_annual_report_details(nse_symbol)
        return details[0] if details else None

    def download_by_ticker(self, nse_symbol: str) -> bool:
        """
        Fully generalized workflow: Looks up the asset link dynamically and downloads it safely.
        """
        report_details = self._get_annual_report_details(nse_symbol)
        if not report_details:
            return False
        pdf_url, report_text = report_details
            
        company_dir = self.download_dir / nse_symbol
        company_dir.mkdir(exist_ok=True)
        year_match = re.search(r"\b(\d{4})\b", report_text)
        year_suffix = f"_{year_match.group(1)}" if year_match else ""
        file_path = company_dir / f"{nse_symbol}_Annual_Report{year_suffix}.pdf"
        
        print(f"📥 Found document target path. Downloading PDF bytes...")
        try:
            # Stream the file content chunks safely
            pdf_response = requests.get(pdf_url, headers=self.headers, verify=False, stream=True, timeout=60)
            if pdf_response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Safe binary storage complete: {file_path}\n")
                return True
            else:
                print(f"❌ Download endpoint rejected request (Status: {pdf_response.status_code})")
        except Exception as e:
            print(f"❌ File extraction error for {nse_symbol}: {str(e)}")
        return False

if __name__ == "__main__":
    downloader = ScreenerPDFDownloader()
    
    # Try any Indian stock symbol you want here!
    portfolio_targets = ["RELIANCE", "INFY", "TCS"]
    
    for stock in portfolio_targets:
        downloader.download_by_ticker(stock)
