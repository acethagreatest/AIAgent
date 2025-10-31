"""API route handlers."""
import asyncio
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.templating import Jinja2Templates

from .state import state
from .tasks import run_nfl_analysis_task


def setup_routes(app: FastAPI, templates_instance: Jinja2Templates) -> None:
    """Register all routes with the FastAPI app."""
    
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates_instance.TemplateResponse(
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
    async def run_nfl() -> RedirectResponse:
        """Start NFL analysis task."""
        # Only start if not running and no pending task
        if not state.is_running and (not state.current_task or state.current_task.done()):
            # create a background task so we can cancel later
            task = asyncio.create_task(run_nfl_analysis_task())
            state.current_task = task
            state.is_running = True  # Set immediately to prevent double-starts
        return RedirectResponse("/", status_code=303)

    @app.get("/status")
    async def status() -> JSONResponse:
        """Get current analysis status."""
        # Check if task is actually done and sync state
        # Priority: Check actual task state over is_running flag
        if state.current_task is not None:
            try:
                if state.current_task.done():
                    # Task completed, ensure is_running is False and clear task
                    state.is_running = False
                    state.current_task = None  # Clear completed task
                else:
                    # Task exists and is not done, ensure is_running is True
                    state.is_running = True
            except Exception:
                # Task might be invalid, clear it
                state.is_running = False
                state.current_task = None
        else:
            # No task exists, ensure is_running is False
            state.is_running = False
        
        return JSONResponse(
            {
                "running": state.is_running,
                "last_result": state.last_result,
                "last_error": state.last_error,
                "num_recommendations": len(state.last_recommendations or []),
            }
        )

    @app.get("/results")
    async def get_results() -> JSONResponse:
        """Get detailed results from the last analysis."""
        recommendations = state.last_recommendations or []
        
        # Convert recommendations to JSON-serializable format
        results = []
        for rec in recommendations:
            try:
                # Convert Pydantic models to dicts
                if hasattr(rec, 'model_dump'):
                    rec_dict = rec.model_dump()
                elif hasattr(rec, 'dict'):
                    rec_dict = rec.dict()
                else:
                    rec_dict = rec
                results.append(rec_dict)  # type: ignore[arg-type]
            except Exception:
                # Fallback for non-serializable objects
                results.append({"error": "Could not serialize recommendation"})  # type: ignore[arg-type]
        
        return JSONResponse(
            {
                "summary": state.last_result,
                "recommendations": results,
                "total": len(results),  # type: ignore[arg-type]
            }
        )

    @app.get("/logs")
    async def get_logs() -> JSONResponse:
        """Get application logs."""
        # return last 300 lines filtered after last clear
        filtered = [line for ver, line in state.logs if ver == state.log_version]
        return JSONResponse({"lines": filtered[-300:]})

    @app.post("/stop")
    async def stop_run() -> JSONResponse:
        """Stop the current analysis immediately without waiting."""
        try:
            if state.current_task and not state.current_task.done():
                state.append_log("⚠️ Stop button pressed - requesting cancellation...")
                state.should_cancel = True
                state.current_task.cancel()
                # Don't wait - just cancel and return immediately
            state.is_running = False
            return JSONResponse({"ok": True, "message": "Stop request received"})
        except Exception as e:
            # Even if there's an error, mark as stopped
            state.is_running = False
            state.should_cancel = True
            return JSONResponse({"ok": True, "message": f"Stop request received (error: {str(e)})"})

    @app.post("/logs/clear")
    async def clear_logs() -> JSONResponse:
        """Clear all logs via POST."""
        # Clear all existing logs and increment version
        state.logs.clear()
        state.log_version += 1
        return JSONResponse({"ok": True})

    @app.get("/logs/clear")
    async def clear_logs_get() -> RedirectResponse:
        """Clear all logs via GET (redirect)."""
        # Clear all existing logs and increment version
        state.logs.clear()
        state.log_version += 1
        return RedirectResponse("/", status_code=303)

