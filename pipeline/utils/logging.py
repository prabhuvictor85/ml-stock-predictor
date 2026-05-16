"""Structured logging utilities."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("artefacts/nse_local/logs")
_file_handler: logging.FileHandler | None = None

_JSON_FMT = logging.Formatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def configure_log_file(log_path: "Path | str") -> None:
    """
    Redirect all pipeline loggers to log_path.

    Call once from main() right after creating the log file — before feature
    engineering starts.  Module-level get_logger() calls during import will
    have already created loggers pointing at a temp file; this function swaps
    that handler out on every registered pipeline logger.
    """
    global _file_handler
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        new_fh.setFormatter(_JSON_FMT)
        new_fh.setLevel(logging.DEBUG)

        old_fh = _file_handler
        _file_handler = new_fh

        # Swap the old handler for the new one on all existing pipeline loggers
        for name, logger in logging.Logger.manager.loggerDict.items():
            if not isinstance(logger, logging.Logger):
                continue
            if not name.startswith("pipeline"):
                continue
            if old_fh is not None and old_fh in logger.handlers:
                logger.removeHandler(old_fh)
            if new_fh not in logger.handlers:
                logger.addHandler(new_fh)

        if old_fh is not None:
            old_fh.close()
    except Exception:
        pass


def _ensure_file_handler() -> logging.FileHandler | None:
    """Create a fallback timestamped log file if configure_log_file() hasn't been called yet."""
    global _file_handler
    if _file_handler is not None:
        return _file_handler
    try:
        from datetime import datetime
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = _LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _file_handler = logging.FileHandler(log_file, encoding="utf-8")
        _file_handler.setFormatter(_JSON_FMT)
        _file_handler.setLevel(logging.DEBUG)
    except Exception:
        pass
    return _file_handler


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger that writes JSON lines to stdout and run.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSON_FMT)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fh = _ensure_file_handler()
        if fh is not None:
            logger.addHandler(fh)
    return logger
