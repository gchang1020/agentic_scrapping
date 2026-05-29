# adapters/html_table_standard.py
# - this is the adapter for pages that contain standard HTML <table> elements
# - this works for most government, financial, and reference data pages
# - interface contract (all adapters must implement this):
#   * fetch(url: str) which returns list[pd.DataFrame]
#   * each DataFrame is one <table> from the page

import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

FALLBACK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# attempt a GET request, and returns Response or None on HTTP error
def _get(url: str, headers: dict):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r
    except requests.exceptions.HTTPError:
        return None
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout fetching {url}")

# download a webpage and return all <table> elements as DataFrames
# - tries FALLBACK_HEADERS automatically if default headers get a 403
# - returns an empty list if no tables are found
def fetch(url: str) -> list[pd.DataFrame]:
    r = _get(url, HEADERS)
    if r is None:
        # if default headers failed, try fallback before giving up
        r = _get(url, FALLBACK_HEADERS)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")
    result = []

    for tbl in tables:
        rows = []
        for tr in tbl.find_all("tr"): # each row in the table
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])] # each cell in the row
            if cells:
                rows.append(cells)
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
            result.append(df) # rows is now a list of lists

    return result


