"""
overview of CrewAI:
 - you describe WHO does the work (agents with roles, goals, backstories)
 - agents autonomously decide WHEN and HOW to call tools
 - the manager LLM orchestrates between agents
 - you see: Agent, Task, Crew, @tool, context=[...]

crewai_scraper_advanced.py:
 - parse args
 - get_the_path()
 - choose_from_list -> myLLM(provider) -> sites configured in sites-advanced.yaml
 - run_once(sites, crewai_llm) — one crew per site in sites-advanced.yaml
   * build four-agent scraping crew across all sites: Scraper -> Analyst -> Fixer -> Reporter
     > Scraper: tools: fetch_html_tables, fetch_csv
     > Analyst: tools: validate_data — context=[t1]
     > Fixer: tools: retry_fallback_headers, write_alert — context=[t1,t2]
     > Reporter: tools=[] — context=[t1,t2,t3]

   * make_tasks() -> Crew.kickoff() -> manager LLM delegates tasks and loop agents via Process.hierarchical
   * each agent: Thought -> @tool called -> Observation fed back -> Final Answer
   * reports -> 'out-reports/' alerts (if any) -> 'out-alerts/'

adapter pattern:
 - each site in sites-advanced.yaml declares an 'adapter' field
 - folder 'adapters/' contains one module (adapter) per scraping strategy
 - @tool functions delegate to the adapter — no fetch logic ever needs to live in this file
 - to support a new site: add adapters/adapter_name.py, then set 'adapter: "adapter_name' in sites-advanced.yaml

structure:
 - garys_llm.py
 - crewai_scraper_advanced.py
 - sites-advanced.yaml
 - adapters/: __init__.py and [adapter.py]
 
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
from bs4 import BeautifulSoup

import yaml
# pip install crewai
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ensure garys_llm.py is on PYTHONPATH
import garys_llm
from garys_llm import myLLM
from adapters import load_adapter

#--------------------------------
# CONFIG
#--------------------------------
sites_yaml = 'sites-advanced.yaml'

# fallback headers available here for the retry tool
FALLBACK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

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
# ADAPTER BRIDGE
#--------------------------------
# adapter.fetch():
# - called by @tool functions by agents, as the bridge between adapters and agents
# - it then converts the output from each adapter, a list of DataFrames into the text preview format
def _dataframes_to_preview(dfs: list) -> str:
    if not dfs:
        return "NO_TABLES_FOUND — adapter returned no data"
    previews = []
    for i, df in enumerate(dfs):
        previews.append(
            f"Table {i+1}: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"Columns: {list(df.columns)}\n"
            f"{df.head(5).to_string(index=False)}"
        )
    return "\n\n---\n\n".join(previews)

#--------------------------------
# TOOLS (what agents can call)
#--------------------------------
# @tool functions:
# - all are thin wrappers because the real fetch logic lives in 'adapters/'
# - the agent picks which tool to call, afterward, the tool delegates to the adapter

# decorated function:
# - use '@' and a name, where name can be any string
# - agent reads the name and docstring, which needs to be
#   * first line after the definition of the function
#   * use a pair of '"""', but not '#', to mark the docstring, stored as function.__doc__
#
# - the regular call to a function is 'fetch_html_tables'
# - the call to a decorated function is now as 'Fetch HTML tables from a webpage'

# _current_site is set before each crew.kickoff() so the @tool functions knows which site is fetched
_current_site = {}

@tool("Fetch HTML tables from a webpage")
def fetch_html_tables(url: str) -> str:
    """Fetch all <table> elements from a page. Returns a text preview of each table."""
    try:
        adapter = load_adapter(_current_site)
        dfs = adapter.fetch(url)
        return _dataframes_to_preview(dfs)
    except Exception as err:
        return f"ERROR: {err}"

@tool("Download and preview a CSV file")
def fetch_csv(url: str) -> str:
    """Download a CSV from a direct URL. Returns shape, columns, and first 5 rows."""
    try:
        adapter = load_adapter(_current_site)
        dfs = adapter.fetch(url)
        return _dataframes_to_preview(dfs)
    except Exception as err:
        return f"ERROR: {err}"

@tool("Validate scraped data quality")
def validate_data(scraped_text: str) -> str:
    """
    Check a scrape result for quality problems.
    Pass the full output from fetch_html_tables or fetch_csv.
    Returns a JSON report with status, issues, and fix suggestions.
    """
    issues, suggestions = [], []
    if "NO_TABLES_FOUND" in scraped_text:
        issues.append("No tables found on page")
        suggestions.append("Site may render tables with JavaScript — try Playwright")
        suggestions.append("Inspect page source to confirm <table> exists")

    if any(w in scraped_text for w in ["ERROR", "error fetching", "Timeout"]):
        issues.append("Fetch failed")
        suggestions.append("Verify URL is still valid and accessible")
        suggestions.append("For 403: rotate User-Agent or add session cookies")

    if "0 rows" in scraped_text:
        issues.append("Table has zero data rows")
        suggestions.append("Header row may be parsed as data — check column detection")

    if scraped_text.count("Unnamed") > 2:
        issues.append("Many unnamed columns — headers likely missing")
        suggestions.append("Use pd.read_csv(url, header=None) and assign names manually")

    status = "OK" if not issues else ("WARNING" if len(issues) == 1 else "FAIL")
    return json.dumps({"status": status, "issues": issues, "fix_suggestions": suggestions}, indent=2)

@tool("Retry fetch with fallback headers")
def retry_with_fallback_headers(url: str) -> str:
    """
    Retry fetching a URL using alternative browser headers.
    Use this when the default fetch returns a 403 or empty result.
    Returns the same format as fetch_html_tables or fetch_csv.
    """
    try:
        r = requests.get(url, headers=FALLBACK_HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        tables = soup.find_all("table")
        if tables:
            rows = []
            for tr in tables[0].find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
                if cells:
                    rows.append(cells)
            if rows:
                df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
                return (f"RETRY SUCCESS — fallback headers worked.\n"
                        f"Table 1: {df.shape[0]} rows x {df.shape[1]} cols\n"
                        f"{df.head(5).to_string(index=False)}")
        df = pd.read_csv(io.StringIO(r.text))
        return (f"RETRY SUCCESS — CSV with fallback headers.\n"
                f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
                f"Columns: {list(df.columns)}")
    except Exception as err:
        return f"RETRY FAILED: {err}"

@tool("Write an alert file")
def write_alert(message: str) -> str:
    """
    Write an alert to out-alerts/ when a problem cannot be fixed automatically.
    Use this as a last resort after retry attempts have failed.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(ALERT_DIR) / f"alert_{ts}.txt"
    path.write_text(f"ALERT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{message}", encoding="utf-8")
    print(f"\n{'!'*60}\nALERT written: {path}\n{message}\n{'!'*60}\n")
    return f"Alert saved to {path}"

#--------------------------------
# AGENTS - built once, reused across all sites in a run
#--------------------------------
# tools = []: to register the actual function objects with the agent
def make_crew(crewai_llm):
    scraper = Agent(
        role = "Web Scraper",
        goal = "Fetch structured data from a URL using the correct tool for its type.",
        backstory = (
            "You specialise in extracting structured data from websites and CSV files. "
            "You always pick the right tool — fetch_csv for .csv URLs, "
            "fetch_html_tables for web pages — and report exactly what you find."
        ),
        tools = [fetch_html_tables, fetch_csv],
        llm = crewai_llm,
        verbose = True,
    )

    analyst = Agent(
        role = "Data Quality Analyst",
        goal = "Validate scraped data and flag anomalies with specific fix proposals.",
        backstory = (
            "You have seen every scraping failure: 403s, JS-rendered tables, "
            "misaligned CSV columns, empty pages. You catch problems early "
            "and always explain how to fix them in concrete Python code."
        ),
        tools = [validate_data],
        llm = crewai_llm,
        verbose = True,
    )

    # fixer: tries to resolve problems before giving up and alerting the user
    fixer = Agent(
        role = "Scrape Fixer",
        goal = (
            "Attempt to fix scraping problems identified by the analyst. "
            "First retry with fallback headers. If that also fails, write an alert "
            "so the user knows manual intervention is needed."
        ),
        backstory = (
            "You are a resilient engineer who never gives up without trying. "
            "When a scrape fails you try alternative approaches before escalating. "
            "You only write an alert when you have genuinely exhausted your options."
        ),
        tools = [retry_with_fallback_headers, write_alert],
        llm = crewai_llm,
        verbose = True,
    )

    reporter = Agent(
        role = "Report Writer",
        goal = "Write a concise, actionable 5-section report from the scrape, analysis, and any fixes.",
        backstory = (
            "You turn raw findings into clear summaries. Your reports are precise, "
            "non-redundant, and always end with a concrete recommendation."
        ),
        tools = [],
        llm = crewai_llm,
        verbose = True,
    )

    return scraper, analyst, fixer, reporter

#--------------------------------
# TASKS
#--------------------------------
# context = [t1]: before running t2, take t1's output and include it in the analyst's prompt
# context = [t1, t2, t3]: receives all three prior outputs
def make_tasks(site, scraper, analyst, fixer, reporter):
    url, name, kind, desc = site["url"], site["name"], site["type"], site["description"]
    tool_instruction = (
        f"Use 'Download and preview a CSV file' with url='{url}'"
        if kind == "csv" else
        f"Use 'Fetch HTML tables from a webpage' with url='{url}'"
    )

    t1 = Task(
        description = (
            f"Fetch data from: {name}\nURL: {url}\nAbout: {desc}\n\n"
            f"{tool_instruction}\n\n"
            "Report: fetch status, table/row count, column names, 5-row preview."
        ),
        expected_output = "Fetch status, shape, column names, and a 5-row data preview.",
        agent = scraper,
    )

    t2 = Task(
        description = (
            "Pass the full scrape output to 'Validate scraped data quality'.\n"
            "Then explain in plain English: status, each issue found, and the fix."
        ),
        expected_output = "Quality status (OK/WARNING/FAIL), issues list, and code fixes.",
        agent = analyst,
        context = [t1],
    )

    t3 = Task(
        description = (
            "Review the quality analysis from the previous task.\n"
            "If status is OK: respond 'No fixes needed.'\n"
            "If status is WARNING or FAIL:\n"
            f"  1. Use 'Retry fetch with fallback headers' with url='{url}'\n"
            "  2. If retry succeeds: report what changed\n"
            "  3. If retry also fails: use 'Write an alert file' describing "
            "     what failed and what was tried, so the user can investigate manually."
        ),
        expected_output = (
            "'No fixes needed' if data was OK. "
            "Otherwise: retry result (success or failure) and alert status."
        ),
        agent = fixer,
        context = [t1, t2],
    )

    t4 = Task(
        description = (
            f"Write a 5-section report for: {name}\n\n"
            "1. DATASET OVERVIEW — what is this data? (2-3 sentences)\n"
            "2. KEY FIELDS — important columns with one-line descriptions\n"
            "3. QUALITY VERDICT — OK / WARNING / FAIL with reason\n"
            "4. FIXES APPLIED — what the fixer tried and whether it worked\n"
            "5. NEXT STEPS — what should happen now"
        ),
        expected_output = "A clean 5-section report, concise and actionable.",
        agent = reporter,
        context = [t1, t2, t3],
    )

    return [t1, t2, t3, t4]

#--------------------------------
# RUNNER
#--------------------------------
# crew.kickoff(): what starts to happen step by step
# - Task() is taken to build LLM prompt, and they chain up w/ each other by definitions
# - in turn, LLM reasons and decides to call a (decorated) tool
# - result goes back to the LLM internally (visible as 'Observation:' in verbose output)
# - LLM decides it's done and produces a final answer, or calls another tool
#
# Process.hierarchical vs Process.sequential:
# - sequential: tasks run in fixed order, no re-delegation
# - hierarchical: a manager LLM oversees the crew, can re-delegate tasks and loop agents back if output dissatisfactory
def run_once(sites, crewai_llm):
    global _current_site
    scraper, analyst, fixer, reporter = make_crew(crewai_llm)
    reports = []

    for site in sites:
        # set _current_site so @tool functions know which adapter to load
        _current_site = site
        print(f"\n{'='*60}\n  {site['name']}\n  {site['url']}")
        print(f"  adapter: {site.get('adapter', 'NOT SET')}\n{'='*60}")

        try:
            tasks  = make_tasks(site, scraper, analyst, fixer, reporter)
            result = Crew(
                # agents does not define execution order, just the existence of agents
                agents = [scraper, analyst, fixer, reporter],
                tasks = tasks,
                # process = Process.sequential,
                process = Process.hierarchical,

                # when set process=Process.hierarchical:
                # - CrewAI automatically creates an invisible manager agent behind the scenes
                # - that manager agent, 'manager_llm', needs an LLM to think with
                manager_llm = crewai_llm,
                verbose = True,
            ).kickoff()
            reports.append(f"# {site['name']}\n\n{result}\n")

        except Exception as err:
            reports.append(f"# {site['name']}\n\nERROR: {err}\n")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"report_{ts}.txt")
    Path(out).write_text(("\n" + "="*60 + "\n\n").join(reports), encoding="utf-8")
    print(f"\nSaved: {out}")

# run repeatedly every interval_minutes. Ctrl+C to stop
def run_loop(sites, interval_minutes, crewai_llm, run_once_only=False):
    print(f"Scheduler started — running every {interval_minutes} min. Ctrl+C to stop.\n")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run...")
        try:
            run_once(sites, crewai_llm)
        except Exception as err:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(ALERT_DIR) / f"alert_crash_{ts}.txt"
            path.write_text(f"CRASH at {datetime.now()}\n{err}", encoding="utf-8")
            print(f"Run crashed — alert written to {path}")

        if run_once_only:
            break
        print(f"Sleeping {interval_minutes} min until next run...")
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    # create an empty parser object
    #parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser(
        description = "CrewAI web scraper — hierarchical multi-agent crew with adapter pattern",
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

    # add arguments: python crewai_scraper_advanced.py --help
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
    Path(OUT_DIR).mkdir(exist_ok=True)
    Path(ALERT_DIR).mkdir(exist_ok=True)

    # get the model
    providers = list(garys_llm.CONFIG.keys())
    idx = choose_from_list("Select AI provider:", providers)
    provider = providers[idx]
    print(f"\nModel: {provider} ({garys_llm.CONFIG[provider]['model']})")

    # load the model
    llm = myLLM(provider)
    crewai_llm = llm.to_crewai()

    if args.loop:
        run_loop(load_sites(args.site), args.loop, crewai_llm, run_once_only=args.once)
    else:
        run_once(load_sites(args.site), crewai_llm)
