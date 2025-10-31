import os
import sys
import time
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import asyncio
from typing import Any, Dict, List

from sports_betting_aggregator import SportsBettingAggregator
from src.flare_ai_kit.sports.models import Sport


if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="AI Agent Dashboard")

app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")


class AppState:
    def __init__(self) -> None:
        self.is_running: bool = False
        self.last_result: Dict[str, Any] | None = None
        self.last_recommendations: List[Any] | None = None
        self.last_error: str | None = None
        self.current_task: asyncio.Task | None = None
        # store (version, line) so we can filter after clear deterministically
        self.logs: List[tuple[int, str]] = []
        self.log_version: int = 0

    def append_log(self, text: str) -> None:
        for line in text.splitlines():
            self.logs.append((self.log_version, line))
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]


state = AppState()


async def run_nfl_analysis_task() -> None:
    state.is_running = True
    state.last_error = None
    state.last_result = None
    state.last_recommendations = None
    try:
        state.append_log("Starting NFL analysis...")
        aggregator = SportsBettingAggregator()
        await aggregator.initialize()
        state.append_log("Initialization complete.")
        data = await aggregator.collect_sports_data([Sport.NFL])
        state.append_log(f"Collected players={len(data.get('players', []))}, games={len(data.get('games', []))}, props={len(data.get('prop_bets', []))}")
        recs = await aggregator.analyze_and_predict(data)
        state.last_result = {
            "players": len(data.get("players", [])),
            "games": len(data.get("games", [])),
            "prop_bets": len(data.get("prop_bets", [])),
        }
        state.last_recommendations = recs
    except Exception as exc:  # noqa: BLE001
        state.last_error = str(exc)
        state.append_log(f"Error: {state.last_error}")
    finally:
        state.is_running = False
        state.current_task = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "is_running": state.is_running,
            "last_result": state.last_result,
            "last_error": state.last_error,
            "recommendations": state.last_recommendations or [],
        },
    )


@app.post("/run/nfl")
async def run_nfl(background_tasks: BackgroundTasks) -> RedirectResponse:
    if not state.is_running:
        # create a background task so we can cancel later
        task = asyncio.create_task(run_nfl_analysis_task())
        state.current_task = task
    return RedirectResponse("/", status_code=303)


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse(
        {
            "running": state.is_running,
            "last_result": state.last_result,
            "last_error": state.last_error,
            "num_recommendations": len(state.last_recommendations or []),
        }
    )


@app.get("/logs")
async def get_logs() -> JSONResponse:
    # return last 300 lines filtered after last clear
    filtered = [line for ver, line in state.logs if ver == state.log_version]
    return JSONResponse({"lines": filtered[-300:]})


@app.post("/stop")
async def stop_run() -> RedirectResponse:
    if state.current_task and not state.current_task.done():
        state.current_task.cancel()
        state.append_log("Cancellation requested.")
    state.is_running = False
    return RedirectResponse("/", status_code=303)


@app.post("/logs/clear")
async def clear_logs() -> JSONResponse:
    state.log_version += 1
    return JSONResponse({"ok": True})


@app.get("/logs/clear")
async def clear_logs_get() -> RedirectResponse:
    state.log_version += 1
    return RedirectResponse("/", status_code=303)


