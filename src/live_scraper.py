"""
Live Web Scraper for ROCm-Pilot.
Fetches and parses official AMD ROCm documentation and PyTorch wheel indexes
to build up-to-date hardware-software compatibility databases.
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger("rocm_pilot.live_scraper")

# Try importing bs4, handle gracefully if missing
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logger.warning("BeautifulSoup4 not installed. HTML parsing functions will be limited.")


class AMDDocsScraper:
    """Scrapes AMD and PyTorch pages for ROCm compatibility data."""

    URLS = {
        'system_requirements': 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html',
        'pytorch_install': 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html',
        'install_guide': 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/',
        'compatibility_matrix': 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/compatibility-matrix.html',
    }

    def __init__(self, cache_dir: str = 'data/scrape_cache', cache_ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_days * 86400

        # Set up Session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def _fetch_page(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetches page content with disk caching, retry, and TTL checks."""
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        cache_path = self.cache_dir / f"{url_hash}.html"
        meta_path = self.cache_dir / f"{url_hash}.meta"

        if use_cache and cache_path.exists() and meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                age = time.time() - meta.get('timestamp', 0)
                if age < self.cache_ttl:
                    logger.debug("Loading %s from cache", url)
                    with open(cache_path, 'r', encoding='utf-8') as cf:
                        return cf.read()
            except Exception as e:
                logger.warning("Error reading cache for %s: %s", url, e)

        # Cache miss or expired
        logger.info("Fetching %s from web...", url)
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                html = response.text
                
                # Check for changes if cached file exists
                if cache_path.exists():
                    with open(cache_path, 'r', encoding='utf-8') as cf:
                        old_html = cf.read()
                    if old_html != html:
                        logger.info("Content changed for %s", url)

                # Write cache
                with open(cache_path, 'w', encoding='utf-8') as cf:
                    cf.write(html)
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump({'timestamp': time.time(), 'url': url}, mf)
                
                return html
            else:
                logger.warning("Failed to fetch %s, status code: %d", url, response.status_code)
                return None
        except Exception as e:
            logger.error("Error fetching %s: %s", url, e)
            return None

    def _parse_html_tables(self, html: str, keyword_filter: Optional[str] = None) -> List[Dict]:
        """Parses HTML tables using BeautifulSoup4, returning structure."""
        if not BeautifulSoup:
            logger.error("BeautifulSoup4 not available. Cannot parse tables.")
            return []

        soup = BeautifulSoup(html, 'lxml' if 'lxml' in globals() else 'html.parser')
        tables = []

        for idx, table_el in enumerate(soup.find_all('table')):
            # Check headers
            headers = []
            thead = table_el.find('thead')
            if thead:
                headers = [th.get_text().strip() for th in thead.find_all('th')]
            else:
                first_tr = table_el.find('tr')
                if first_tr:
                    headers = [th.get_text().strip() for th in first_tr.find_all(['th', 'td'])]

            # Apply keyword filter
            if keyword_filter and not any(keyword_filter.lower() in h.lower() for h in headers):
                # Check table body text just in case
                if keyword_filter.lower() not in table_el.get_text().lower():
                    continue

            # Parse rows
            rows = []
            tbody = table_el.find('tbody')
            tr_elements = tbody.find_all('tr') if tbody else table_el.find_all('tr')[1:]
            for tr in tr_elements:
                cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)

            if headers or rows:
                tables.append({
                    'index': idx,
                    'headers': headers,
                    'rows': rows,
                    'raw_html': str(table_el)
                })

        return tables

    def scrape_system_requirements(self) -> Dict:
        """Parses GPU support matrix and kernel requirements from system requirements page."""
        html = self._fetch_page(self.URLS['system_requirements'])
        if not html:
            return {}

        tables = self._parse_html_tables(html, keyword_filter='gfx')
        # If no explicit gfx table, try 'GPU' or 'architecture'
        if not tables:
            tables = self._parse_html_tables(html, keyword_filter='GPU')

        gpu_data = {}
        for table in tables:
            headers = [h.lower() for h in table['headers']]
            
            # Map column indices
            gpu_idx = -1
            gfx_idx = -1
            family_idx = -1

            for col_idx, header in enumerate(headers):
                if 'gpu' in header or 'product' in header or 'model' in header:
                    gpu_idx = col_idx
                elif 'target' in header or 'gfx' in header or 'llvm' in header or 'id' in header:
                    gfx_idx = col_idx
                elif 'family' in header or 'architecture' in header:
                    family_idx = col_idx

            if gfx_idx != -1:
                # We have a valid hardware matrix
                for row in table['rows']:
                    if len(row) <= max(gpu_idx, gfx_idx):
                        continue
                    
                    gfx_raw = row[gfx_idx]
                    gpu_name = row[gpu_idx] if gpu_idx != -1 else ""
                    family = row[family_idx] if family_idx != -1 else ""

                    # Find gfx tags like gfx906, gfx942
                    matches = re.findall(r'gfx\d{3,4}', gfx_raw.lower())
                    for gfx in matches:
                        gpu_data[gfx] = {
                            'gfx_id': gfx,
                            'name': gpu_name.split('(')[0].strip(),
                            'family': family.strip(),
                            'source': 'scraped_requirements'
                        }
        
        logger.info("Scraped %d GPU architectures from system requirements", len(gpu_data))
        return gpu_data

    def scrape_pytorch_compatibility(self) -> Dict:
        """Extracts pip install and docker information from PyTorch installation docs."""
        html = self._fetch_page(self.URLS['pytorch_install'])
        if not html:
            return {}

        # Look for pip install code blocks using regex
        pip_installs = {}
        # pip install pattern: pip3? install torch... --index-url ...whl/rocmX.Y
        pattern = r'pip3?\s+install\s+torch[^\n]+whl/rocm\d\.\d'
        matches = re.findall(pattern, html)
        for match in matches:
            # Strip html tags if present
            cleaned = re.sub('<[^<]+?>', '', match).strip()
            # Extract ROCm version
            rocm_match = re.search(r'rocm(\d+\.\d+)', cleaned)
            if rocm_match:
                rocm_ver = rocm_match.group(1)
                pip_installs[rocm_ver] = cleaned

        logger.info("Scraped %d ROCm pip install commands", len(pip_installs))
        return pip_installs

    def scrape_pytorch_wheel_index(self) -> Dict[str, List[str]]:
        """Scrapes pytorch.org wheel directory to verify which pytorch versions support ROCm."""
        # Query download.pytorch.org wheel index
        url = 'https://download.pytorch.org/whl/'
        html = self._fetch_page(url)
        if not html:
            return {}

        # Find all rocmX.Y subdirs
        rocm_dirs = re.findall(r'rocm(\d+\.\d+)/', html)
        rocm_dirs = sorted(list(set(rocm_dirs)))

        rocm_to_torch = {}
        for rocm_ver in rocm_dirs:
            # We don't want to crawl all nested whl files to avoid heavy traffic.
            # Instead, we just document that ROCm subdirectory exists.
            rocm_to_torch[rocm_ver] = []
            
        logger.info("Detected %d ROCm wheel subdirectories on download.pytorch.org", len(rocm_to_torch))
        return rocm_to_torch

    def scrape_radeon_wheel_index(self) -> Dict[str, List[str]]:
        """Scrapes repo.radeon.com wheel directories to map release versions."""
        url = 'https://repo.radeon.com/rocm/manylinux/'
        html = self._fetch_page(url)
        if not html:
            return {}

        # Find release folders like rocm-rel-6.1/
        rel_dirs = re.findall(r'rocm-rel-(\d+\.\d+(?:\.\d+)?)/', html)
        rel_dirs = sorted(list(set(rel_dirs)))
        
        logger.info("Detected %d ROCm release wheel directories on repo.radeon.com", len(rel_dirs))
        return {ver: [] for ver in rel_dirs}

    def scrape_all(self) -> Dict:
        """Executes all scraping stages and rolls results into a single dict."""
        start_time = time.time()
        logger.info("Beginning full live scraping workflow...")

        results = {
            'timestamp': time.time(),
            'gpus': self.scrape_system_requirements(),
            'pytorch_compat': self.scrape_pytorch_compatibility(),
            'pytorch_whl_dirs': self.scrape_pytorch_wheel_index(),
            'radeon_whl_dirs': self.scrape_radeon_wheel_index(),
            'elapsed_sec': time.time() - start_time
        }
        return results

    def update_gpu_database(self, db_path: str = 'data/gpu_database.json') -> Dict:
        """
        Loads the existing GPU database, updates it with live scraped compatibility info,
        merging new features while preserving manual entries and notes.
        """
        # Load local database
        path = Path(db_path)
        if not path.exists():
            # Fallback relative to script
            path = Path(__file__).parent.parent / db_path

        if not path.exists():
            logger.warning("Local GPU database not found at %s. Creating new seed database.", db_path)
            gpu_db = {
                "last_updated": "",
                "source_urls": self.URLS,
                "gpu_architectures": {},
                "rocm_versions": {},
                "pytorch_rocm_matrix": {}
            }
        else:
            with open(path, 'r', encoding='utf-8') as f:
                gpu_db = json.load(f)

        # Run scraper
        scraped = self.scrape_all()

        # Update GPU architectures
        local_archs = gpu_db.setdefault("gpu_architectures", {})
        for gfx_id, live_gpu in scraped['gpus'].items():
            if gfx_id in local_archs:
                # Merge: only update if missing or empty
                local_gpu = local_archs[gfx_id]
                if not local_gpu.get("name"):
                    local_gpu["name"] = live_gpu["name"]
                if not local_gpu.get("family"):
                    local_gpu["family"] = live_gpu["family"]
            else:
                # Add new scraped GPU architecture
                local_archs[gfx_id] = {
                    "gfx_id": gfx_id,
                    "name": live_gpu["name"],
                    "family": live_gpu["family"],
                    "category": "consumer" if "rx" in live_gpu["name"].lower() or "radeon" in live_gpu["name"].lower() else "datacenter",
                    "vram_gb": [16],
                    "min_rocm": "6.0",
                    "max_rocm": "",
                    "rocm_versions": [],
                    "pytorch_versions": [],
                    "status": "supported",
                    "notes": "Added from live scraping."
                }

        # Update PyTorch ROCm matrix
        local_matrix = gpu_db.setdefault("pytorch_rocm_matrix", {})
        for rocm_ver, pip_cmd in scraped['pytorch_compat'].items():
            # Estimate PyTorch version from pip command
            # e.g., "torch==2.5.1+rocm6.2" -> "2.5"
            pt_match = re.search(r'torch==(\d+\.\d+)', pip_cmd)
            if pt_match:
                pt_ver = pt_match.group(1)
                
                # Get or create matrix entry
                pt_entry = local_matrix.setdefault(pt_ver, {
                    "rocm_versions": [],
                    "pip_install": "",
                    "docker_image": ""
                })
                
                if rocm_ver not in pt_entry["rocm_versions"]:
                    pt_entry["rocm_versions"].append(rocm_ver)
                pt_entry["pip_install"] = pip_cmd

        # Save back to disk
        gpu_db["last_updated"] = time.strftime("%Y-%m-%d")
        gpu_db["source_urls"] = self.URLS

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gpu_db, f, indent=2)

        logger.info("Updated GPU compatibility database at %s", path)
        return gpu_db

    def validate_urls(self, urls: List[str]) -> Dict[str, Dict]:
        """HEAD check a list of URLs to verify they are reachable."""
        validation = {}
        for url in urls:
            try:
                resp = self.session.head(url, timeout=5, allow_redirects=True)
                validation[url] = {
                    'reachable': resp.status_code < 400,
                    'status_code': resp.status_code,
                    'redirect_url': resp.url if resp.url != url else None
                }
            except Exception as e:
                validation[url] = {
                    'reachable': False,
                    'error': str(e)
                }
        return validation


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    scraper = AMDDocsScraper()
    
    print("Testing live scraper functions...")
    scraped_gpus = scraper.scrape_system_requirements()
    print(f"Scraped GPUs keys: {list(scraped_gpus.keys())}")
    
    # Run database update test
    db = scraper.update_gpu_database()
    print("Database update test complete. Last updated:", db.get("last_updated"))
