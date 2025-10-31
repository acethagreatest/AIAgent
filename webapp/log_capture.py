"""Log capture utilities for redirecting stdout/stderr to application logs."""
import sys
from typing import Any, Callable


class LogCaptureWriter:
    """A file-like object that redirects writes to AppState logs."""
    
    def __init__(self, append_func: Callable[[str], None]) -> None:
        self.append_func = append_func
        self.buffer = ""
    
    def write(self, text: str) -> int:
        """Buffer text and flush on newlines."""
        text = str(text)
        self.buffer += text
        # Flush complete lines
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():  # Only log non-empty lines
                self.append_func(line + '\n')
        return len(text)
    
    def flush(self) -> None:
        """Flush any remaining buffer."""
        if self.buffer.strip():
            self.append_func(self.buffer)
            self.buffer = ""


class LogCapture:
    """Capture stdout/stderr and redirect to AppState logs in real-time."""
    
    def __init__(self, append_func: Callable[[str], None]) -> None:
        self.append_func = append_func
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.stdout_writer = LogCaptureWriter(append_func)
        self.stderr_writer = LogCaptureWriter(append_func)
        # Also capture Python logging
        self.old_logging_handler: Any = None
        self.logging_handler: Any = None
        
    def __enter__(self):
        """Context manager entry."""
        sys.stdout = self.stdout_writer
        sys.stderr = self.stderr_writer
        
        # Also capture Python logging (which structlog uses)
        import logging
        self.logging_handler = logging.StreamHandler(self.stdout_writer)
        self.logging_handler.setLevel(logging.DEBUG)
        # Get root logger and add our handler
        root_logger = logging.getLogger()
        if root_logger.handlers:
            self.old_logging_handler = root_logger.handlers[0]
        root_logger.addHandler(self.logging_handler)
        root_logger.setLevel(logging.DEBUG)
        
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        # Restore logging
        import logging
        root_logger = logging.getLogger()
        if self.logging_handler:
            root_logger.removeHandler(self.logging_handler)
        if self.old_logging_handler:
            root_logger.addHandler(self.old_logging_handler)
        
        return False

