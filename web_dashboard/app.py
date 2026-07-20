import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
load_dotenv()

app = FastAPI(title="Trading AI Dashboard")
security = HTTPBasic()

# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Create static dir if not exists
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Auth
def get_current_username(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
    correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"),
        os.getenv("DASHBOARD_USER", "admin").encode("utf8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"),
        os.getenv("DASHBOARD_PASS", "admin").encode("utf8")
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username



def get_recent_logs(lines=50):
    log_files = ["main.log", "trading_journal.csv", "morning_brief/output/morning_brief.log"]
    logs = []

    for f in log_files:
        path = Path(BASE_DIR.parent) / f
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as file:
                    file_content = file.readlines()
                    # Append up to 'lines' lines PER file, instead of truncating all globally.
                    logs.extend([f"[{f}] {line.strip()}" for line in file_content[-lines:] if line.strip()])
            except Exception as e:
                logs.append(f"Erreur lecture {f}: {e}")

    # If we want the *absolute* last 'lines' logs across all files, we can sort by date/time if present.
    # Otherwise we just return them all so we don't truncate older files silently.
    return logs

@app.get("/")
def read_root(request: Request, username: Annotated[str, Depends(get_current_username)]):
    logs = get_recent_logs()
    return templates.TemplateResponse(request=request, name="index.html", context={"username": username, "logs": logs})

import sqlite3
import pandas as pd

def get_db_connection():
    db_path = Path(BASE_DIR.parent) / "trading_history.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/performance")
def read_performance(request: Request, username: Annotated[str, Depends(get_current_username)]):
    # Read performance data from database
    db_path = Path(BASE_DIR.parent) / "trading_history.db"

    portfolio_history = []
    transactions = []

    if db_path.exists():
        try:
            conn = get_db_connection()
            # Try to get portfolio history
            try:
                df_port = pd.read_sql_query("SELECT * FROM portfolio_history ORDER BY timestamp DESC LIMIT 100", conn)
                portfolio_history = df_port.to_dict('records')
            except Exception as e:
                print(f"Error reading portfolio_history: {e}")

            # Try to get transactions
            try:
                df_trans = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC LIMIT 100", conn)
                transactions = df_trans.to_dict('records')
            except Exception as e:
                print(f"Error reading transactions: {e}")

            conn.close()
        except Exception as e:
            print(f"Database error: {e}")

    return templates.TemplateResponse(request=request, name="performance.html", context={
        "request": request,
        "username": username,
        "portfolio_history": portfolio_history,
        "transactions": transactions
    })

@app.get("/reports")
def read_reports(request: Request, username: Annotated[str, Depends(get_current_username)]):
    morning_brief_dir = Path(BASE_DIR.parent) / "morning_brief" / "output"
    council_dir = Path(BASE_DIR.parent) / "docs" / "council_reports"

    morning_reports = []
    if morning_brief_dir.exists():
        for file in sorted(morning_brief_dir.glob("*.md"), reverse=True):
            with open(file, "r", encoding="utf-8") as f:
                morning_reports.append({
                    "name": file.name,
                    "content": f.read()
                })

    council_reports = []
    if council_dir.exists():
        for file in sorted(council_dir.glob("*.md"), reverse=True):
            with open(file, "r", encoding="utf-8") as f:
                council_reports.append({
                    "name": file.name,
                    "content": f.read()
                })

    return templates.TemplateResponse(request=request, name="reports.html", context={
        "request": request,
        "username": username,
        "morning_reports": morning_reports,
        "council_reports": council_reports
    })

from src.t212_executor import load_portfolio_state, get_t212_positions

@app.get("/positions")
def read_positions(request: Request, username: Annotated[str, Depends(get_current_username)]):
    # Fetch local state
    try:
        local_state_qqq = load_portfolio_state("QQQ", sync=False)
    except Exception as e:
        local_state_qqq = {"error": str(e)}

    try:
        local_state_wti = load_portfolio_state("CL=F", sync=False)
    except Exception as e:
        local_state_wti = {"error": str(e)}

    # Fetch T212 real positions
    try:
        t212_positions = get_t212_positions()
    except Exception as e:
        t212_positions = {"error": str(e)}

    return templates.TemplateResponse(request=request, name="positions.html", context={
        "request": request,
        "username": username,
        "local_state_qqq": local_state_qqq,
        "local_state_wti": local_state_wti,
        "t212_positions": t212_positions
    })
