"""Application state management."""
import asyncio
from typing import Any, Dict, List


class AppState:
    """Global application state for the dashboard."""
    
    def __init__(self) -> None:
        self.is_running: bool = False
        self.last_result: Dict[str, Any] | None = None
        self.last_recommendations: List[Any] | None = None
        self.last_error: str | None = None
        self.current_task: asyncio.Task[None] | None = None
        # store (version, line) so we can filter after clear deterministically
        self.logs: List[tuple[int, str]] = []
        self.log_version: int = 0
        self.should_cancel: bool = False

    def append_log(self, text: str) -> None:
        """Append a log entry to the logs list."""
        for line in text.splitlines():
            if line.strip():  # Only add non-empty lines
                self.logs.append((self.log_version, line))
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
    
    def check_cancelled(self) -> None:
        """Check if cancellation was requested and raise CancelledError if so."""
        if self.should_cancel:
            raise asyncio.CancelledError("Cancellation requested")


# Global state instance
state = AppState()

