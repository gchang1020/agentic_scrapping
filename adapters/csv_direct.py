# adapters/csv_direct.py
# - adapter for direct CSV file URLs
# - this works for any URL that returns a raw CSV file when fetched
# - interface contract (all adapters must implement this):
#   * fetch(url: str) which returns list[pd.DataFrame]
#   * the returned single-element list is for the consistency with html_table_standard

import io
import requests
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/csv,text/plain,*/*;q=0.8",
}

# download a CSV from a direct URL and return as a single-element list
def fetch(url: str) -> list[pd.DataFrame]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return [df]

    except requests.exceptions.HTTPError as err:
        code = err.response.status_code
        hint = "Fix: add session cookies or find an API endpoint" if code == 403 else f"HTTP {code}"
        raise RuntimeError(f"{code} error fetching {url}\n{hint}")

    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout fetching {url}")
