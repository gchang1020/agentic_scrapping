"""
overview LangChain:
 - you describe WHAT happens and in what order (a pipeline of steps)
 - you as an orchestrator decide when each step runs — no autonomous decisions
 - the LLM is called directly inside steps, not by a framework agent
 - you see: RunnableLambda, the "|" pipe operator, plain functions

langchain_scraper.py
 - parse args
 - get_the_path
 - choose_from_list -> myLLM(provider): sites configured in sites-simple.yaml
 - run_once(sites, llm)
   * build_chain(llm): four-step pipe chain built as fetch -> validate -> summarise -> fix_proposals
   * go through each step: chain.invoke(site)
   * return state dict by format_result() and save to file for the run

structure
 - garys_llm.py
 - langchain_scraper_advanced.py
 
usage:
 - python crewai_scraper.py # same as --help
 - python crewai_scraper.py --once # run once
 - python crewai_scraper.py --site "FDIC" # filter by name
 - python crewai_scraper.py --loop 60 # run every 60 minutes
 - python crewai_scraper.py --site "FDIC" --loop 30
 - python crewai_scraper.py --site "FDIC" --loop 30 --once
"""

# pip install requests beautifulsoup4 pandas lxml pyyaml
import io, os, sys, json, time, argparse, requests, pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

import yaml
# pip install langchain langchain-core
from langchain_core.runnables import RunnableLambda

# ensure garys_llm.py is on PYTHONPATH
import garys_llm
from garys_llm import myLLM

#--------------------------------
# CONFIG
#--------------------------------
sites_yaml = 'sites-simple.yaml'

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

FALLBACK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

# keyword to be matched in 'name' of each entry in yaml
def load_sites(name_filter=None):
    data  = yaml.safe_load(SITES_FILE.read_text(encoding="utf-8"))
    sites = [s for s in data["sites"] if s.get("enabled", True)]

    # filter the sites
    if name_filter:
        sites = [s for s in sites if name_filter.lower() in s["name"].lower()]
        if not sites:
            sys.exit(f"No site matching '{name_filter}' in '{sites_yaml}'")
    return sites

def get_the_path():
    while True:
        pname = input("Path (not the file) ([Q]uit)? ")
        if pname.lower() == 'q' or pname == '':
            exit()
        elif not os.path.exists(pname) or not os.path.isdir(pname):
            print(f"Path '{pname}' not found")
            continue
        return pname

def choose_from_list(title: str, options: list[str]) -> int:
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f" - {i}: {opt}")

    while True:
        raw = input("Choice? ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"Please enter a number between 1 and {len(options)}")

#--------------------------------
# SCRAPING
#--------------------------------
# FetchError:
# - it inherits everything from Exception
# - instantiating it like "FetchError("...") so that "Exception.__init__(self, "...")
class FetchError(Exception):
    pass

def _fetch(url, headers=None):
    try:
        r = requests.get(url, headers=headers or HEADERS, timeout=20)
        r.raise_for_status()
        return r

    except requests.exceptions.HTTPError as err:
        code = err.response.status_code
        hint = "Fix: add session cookies, use Playwright or find an API endpoint" if code == 403 else f"HTTP {code}"
        raise FetchError(f"{code} error fetching {url}\n{hint}")

    except requests.exceptions.Timeout:
        raise FetchError(f"Timeout fetching {url}")

def fetch_html_tables(url):
    # returns a list of DataFrames, one per <table> on the page
    r = _fetch(url)
    soup = BeautifulSoup(r.text, "lxml")
    dfs = []
    for tbl in soup.find_all("table"):
        rows = []
        for tr in tbl.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
            dfs.append(df)
    return dfs

def fetch_csv(url):
    # returns a single DataFrame
    return pd.read_csv(io.StringIO(_fetch(url).text))

def fetch_with_fallback(url):
    # retry using alternative browser headers — called in step_fix if step_fetch failed
    return _fetch(url, headers=FALLBACK_HEADERS)

#--------------------------------
# VALIDATION
#--------------------------------
# @dataclass is a decorator, defined in Python standard library
# - it auto-generates __init__, __repr__, and __eq__ based on 'status', 'issues', 'suggestions'
# - unlike CrewAI: YOU define and carry state with no framework does the work

# Check:
# - it a structured carrier:
#   * it is created in validate()
#   * it returns one quality Check per DataFrame or between steps
#   * without it you'd pass raw strings or tuples and lose the named access like c.ok and c.as_text()
#   * it's stored in the state dict, and read by two downstream steps
#
# - it holds the three things those steps need: 
#   * status (for decisions), 
#   * issues (for the LLM prompt),
#   * suggestions (for the fix proposals)
@dataclass
class Check:
    # functions generated:
    # - 'def __init__(self, status, issues, suggestions)' as the constructor
    # - 'def __repr__(self)': so that print(c) will give sth readable
    status: str # "OK", "WARNING", or "FAIL"
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # @property: 
    # - a built-in Python decorator that lets you call a method without parentheses
    # - without it, call c.ok() with parentheses; with it, c.ok can be called
    @property
    def ok(self):
        return self.status == "OK"

    def as_text(self):
        if self.ok: # call c.ok not c.ok()
            return "Status: OK — no issues"
        lines = [f"Status: {self.status}"]
        for i in self.issues:
            lines.append(f"  Issue: {i}")
        for s in self.suggestions:
            lines.append(f"  Fix: {s}")
        return "\n".join(lines)

def validate(df):
    issues, suggestions = [], []

    if df is None:
        return Check("FAIL", ["DataFrame is None"], ["Check whether fetch succeeded"])
    if not isinstance(df, pd.DataFrame):
        return Check("FAIL", [f"Expected DataFrame, got {type(df)}"], ["Check fetch function return type"])
    if df.empty:
        return Check("FAIL", ["DataFrame is empty"], ["Check whether fetch succeeded and source still has data"])

    #print(f"  shape: {df.shape}")
    #print(f"  dtypes:\n{df.dtypes}")
    #print(f"  head:\n{df.head(2)}")
    
    print("  checking row count...")
    if df.shape[0] < 2:
        issues.append("Only 1 data row — partial scrape?")
        suggestions.append("Check for pagination or JS-rendered content")

    print("  checking empty cols...")
    # empty_cols = [c for c in df.columns if bool(df[c].isna().all())]
    empty_cols = [df.columns[i] for i in range(len(df.columns)) if bool(df.iloc[:, i].isna().all())]
    if empty_cols:
        issues.append(f"Entirely empty columns: {empty_cols}")
        suggestions.append("df.dropna(axis=1, how='all')")

    print("  checking unnamed cols...")
    #unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    unnamed = [df.columns[i] for i in range(len(df.columns)) if str(df.columns[i]).startswith("Unnamed")]
    if len(unnamed) > len(df.columns) * 0.4:
        issues.append("Many unnamed columns — headers likely missing")
        suggestions.append("pd.read_csv(url, header=None) then assign column names")

    # check if a row is identical to a prior row one value per row
    print("  checking duplicates...")
    dups = int(df.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate rows")
        suggestions.append("df.drop_duplicates(inplace=True)")

    print("  done")
    if not issues:
        return Check("OK")
    return Check("WARNING" if len(issues) == 1 else "FAIL", issues, suggestions)

#--------------------------------
# CHAIN STEPS
#--------------------------------
# creates the pipeline or the chain once and reused for every site
def build_chain(llm):
    # the | operator connects steps into a pipeline:
    # - output from one step feeds into the next as state dict
    # - this contrast with CrewAI's context=[t1, t2]: here the whole state flows automatically
    return (
        RunnableLambda(step_fetch)
        | RunnableLambda(step_validate)
        | RunnableLambda(step_summarise(llm))
        | RunnableLambda(step_fix(llm))
    )

# chains:
# - each step is a plain function that receives the full state dict and returns it updated
# - contrast with CrewAI: there is no role, no backstory, no autonomous decision — just a function
# - "|" pipe operator in build_chain(): step_fetch -> step_validate -> step_summarise -> step_fix
#   * after step_fetch: state gets 'tables', 'fetch_error'
#   * after step_validate: state gets 'checks', 'any_issues'
#   * after step_summarise: state gets 'summary'
#   * after step_fix: state gets 'fix_proposal' or writes alert file
def step_fetch(state):
    url, kind = state["url"], state["type"]
    print(f"  [fetch] {kind} — {url}")
    try:
        state["tables"] = [fetch_csv(url)] if kind == "csv" else fetch_html_tables(url)
        state["fetch_error"] = None

    except FetchError as err:
        state["tables"] = []
        state["fetch_error"] = str(err)

    return state

def step_validate(state):
    #print("  [validate]")
    #print(f"  fetch_error: {state.get('fetch_error')}")
    #print(f"  tables count: {len(state.get('tables', []))}")
    for i, t in enumerate(state.get('tables', [])):
        print(f"  table[{i}] type: {type(t)}")

    if state.get("fetch_error"):
        # nothing to validate — propagate the error forward
        state["checks"] = []
        state["any_issues"] = True
        return state

    if not state["tables"]:
        state["checks"] = [Check("FAIL", ["No tables found on page"], ["Site may render with JS — try Playwright"])]
        state["any_issues"] = True
        return state

    state["checks"] = [validate(df) for df in state["tables"]]
    state["any_issues"] = any(not c.ok for c in state["checks"])
    return state
    
def step_summarise(llm):
    # contrast with CrewAI: the agent owns its llm, but here YOU pass it in explicitly
    def _run(state):
        print("  [summarise] -> LLM")

        if state.get("fetch_error"):
            state["summary"] = f"Fetch failed: {state['fetch_error']}"
            return state

        parts = [
            f"Source: {state['name']}",
            f"URL: {state['url']}",
            f"Description: {state['description']}",
        ]
        for i, (df, c) in enumerate(zip(state["tables"], state["checks"])):
            preview = (
                f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
                f"Columns: {list(df.columns)}\n"
                f"{df.head(5).to_string(index=False)}"
            )
            parts.append(f"\nTable {i+1}:\n{preview[:600]}\n{c.as_text()}")

        state["summary"] = llm.generate(
            "You are a data analyst. Given this scraped data, write:\n"
            "1. What the dataset contains (2 sentences)\n"
            "2. Most important columns\n"
            "3. Overall quality assessment\n\n"
            + "\n".join(parts)
        )
        return state

    return _run

def step_fix(llm):
    # scenairos:
    # - if data is clean: skip LLM call entirely
    # - if issues found: try fallback fetch first, then ask LLM for fix proposals
    # - if unfixable: write an alert file (same role as the Fixer agent in CrewAI)
    def _run(state):
        # check if there's anything to fix
        if not state.get("any_issues"):
            state["fix_proposal"] = None
            return state

        print("  [fix] issues detected — trying fallback fetch...")

        # step 1: retry with fallback headers (mirrors Fixer agent's retry tool)
        retry_result = None
        try:
            r = fetch_with_fallback(state["url"])
            soup = BeautifulSoup(r.text, "lxml")
            tables = soup.find_all("table")
            if tables:
                retry_result = f"RETRY SUCCESS — fallback headers worked, found {len(tables)} table(s)"
            else:
                df = pd.read_csv(io.StringIO(r.text))
                retry_result = f"RETRY SUCCESS — CSV with fallback headers, {df.shape[0]} rows"

        except Exception as err:
            retry_result = f"RETRY FAILED: {err}"

        # step 2: ask LLM for fix proposals regardless of retry outcome
        print("  [fix] -> LLM for fix proposals")
        if state.get("fetch_error"):
            issues_text = state["fetch_error"]
        else:
            issues_text = "\n".join(f"Table {i+1}:\n{c.as_text()}" for i, c in enumerate(state["checks"]) if not c.ok)

        # ask LLM for fix proposals:
        # - directly, rather than through an agent
        # - fixer agent in CrewAI will decide when to retry, when to alert, and what to say
        state["fix_proposal"] = llm.generate(
            f"A Python scraper fetched: {state['url']}\n\n"
            f"Retry attempt: {retry_result}\n\n"
            f"Problems found:\n{issues_text}\n\n"
            "Give 2-3 specific Python code fixes. Show actual snippets."
        )

        # step 3: write alert file if retry also failed (mirrors Fixer agent's write_alert tool)
        if "RETRY FAILED" in retry_result:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(ALERT_DIR) / f"alert_{ts}.txt"
            msg = (
                f"ALERT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Site: {state['name']}\nURL: {state['url']}\n\n"
                f"Retry result: {retry_result}\n\n"
                f"Issues:\n{issues_text}\n\n"
                f"Fix proposals:\n{state['fix_proposal']}"
            )
            path.write_text(msg, encoding="utf-8")
            print(f"\n{'!'*60}\nALERT written: {path}\n{'!'*60}\n")

        return state

    return _run

#--------------------------------
# RUNNER
#--------------------------------
# how LangChain chain.invoke() works step by step:
# - state dict enters step_fetch -> tables and fetch_error added
# - flows into step_validate -> checks and any_issues added
# - flows into step_summarise -> LLM called directly, summary added
# - flows into step_fix -> LLM called if issues, fix_proposal added
#
# this contrasts with CrewAI's crew.kickoff():
# - no Crew(), no autonomous agent, no tasks, no reasoning here
# - no 'Thought / Action / Observation' loop
# - you control every decision explicitly in the step functions
def format_result(r):
    lines = [f"Source: {r['name']}", f"URL:    {r['url']}", ""]

    if r.get("fetch_error"):
        lines.append(f"FETCH FAILED\n{r['fetch_error']}")
    else:
        for i, c in enumerate(r.get("checks", [])):
            lines.append(f"Table {i+1}: {c.as_text()}")

    lines += ["", f"Summary:\n{r.get('summary', 'n/a')}"]

    if r.get("fix_proposal"):
        lines += ["", f"Fix proposals:\n{r['fix_proposal']}"]

    return "\n".join(lines)

def run_once(sites, llm):
    chain   = build_chain(llm)
    reports = []

    # invoke the chain once per site
    for site in sites:
        print(f"\n{'='*60}\n  {site['name']}\n  {site['url']}\n{'='*60}")
        try:
            result = chain.invoke(site)
            report = format_result(result)
        except Exception as err:
            report = f"ERROR: {err}"

        print(report)
        reports.append(f"# {site['name']}\n\n{report}\n")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"report_{ts}.txt")
    Path(out).write_text(("\n" + "="*60 + "\n\n").join(reports), encoding="utf-8")
    print(f"\nSaved: {out}")

def run_loop(sites, interval_minutes, llm, run_once_only=False):
    """Run repeatedly every interval_minutes. Ctrl+C to stop."""
    print(f"Scheduler started — running every {interval_minutes} min. Ctrl+C to stop.\n")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run...")
        try:
            run_once(sites, llm)
            
        except Exception as err:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(ALERT_DIR) / f"alert_crash_{ts}.txt"
            path.write_text(f"CRASH at {datetime.now()}\n{err}", encoding="utf-8")
            print(f"Run crashed — alert written to {path}")

        if run_once_only:
            break
            
        print(f"Sleeping {interval_minutes} min until next run...")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "LangChain web scraper — explicit step-by-step pipe chain",
        epilog = (
            "examples:\n"
            "  python crewai_scraper.py (same as --help)\n"
            "  python crewai_scraper.py --once\n"
            "  python crewai_scraper.py --site 'FDIC'\n"
            "  python crewai_scraper.py --site 'FDIC' --loop 30\n"
            "  python crewai_scraper.py --site 'FDIC' --loop 30 --once\n"
            "  python crewai_scraper.py --loop 60"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # add arguments: python langchain_scraper.py --help
    parser.add_argument("--site", help=f"Filter by site name based on '{sites_yaml}'")
    parser.add_argument("--loop", type=int, metavar="MINUTES", help="Run every N minutes (omit for single run)")
    parser.add_argument("--once", action="store_true", help="With --loop: run once then stop instead of looping")

    # actually read sys.argv and match against those definitions
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit()

    # get the directory for yaml and output
    pname = get_the_path()
    SITES_FILE = Path(os.path.join(pname, f"{sites_yaml}"))
    OUT_DIR = os.path.join(pname, "out-reports")
    ALERT_DIR = os.path.join(pname, "out-alerts")

    # get the model
    providers = list(garys_llm.CONFIG.keys())
    idx = choose_from_list("Select AI provider:", providers)
    provider = providers[idx]
    print(f"\nModel: {provider} ({garys_llm.CONFIG[provider]['model']})")

    # load the model
    # - contrast with CrewAI: llm is passed directly into run_once/run_loop
    # - no .to_crewai() bridge needed — LangChain steps call llm.generate() directly
    llm = myLLM(provider)

    if args.loop:
        run_loop(load_sites(args.site), args.loop, llm, run_once_only=args.once)
    else:
        run_once(load_sites(args.site), llm)
