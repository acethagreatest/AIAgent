"""Main FastAPI application entry point."""
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

# Configure encoding for Windows
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# Create FastAPI app
app = FastAPI(title="AI Agent Dashboard")

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")
templates = Jinja2Templates(directory="webapp/templates")

# Import and setup routes (must be at end to avoid circular imports)
try:
    from .routes import setup_routes
except ImportError:
    # Fallback for when running as script
    from webapp.routes import setup_routes

setup_routes(app, templates)
