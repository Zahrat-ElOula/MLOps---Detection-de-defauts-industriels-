"""Structured logging configuration.

Application logs are emitted as single-line JSON so they can be shipped to
Loki/ELK without a custom parser. Set ``LOG_JSON=0`` for human-readable logs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` including any custom ``extra`` fields."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str | int | None = None,
    json_logs: bool | None = None,
) -> logging.Logger:
    """Configure the root logger once and return it.

    Parameters
    ----------
    level:
        Logging level name or numeric level. Defaults to ``LOG_LEVEL`` or
        ``"INFO"``.
    json_logs:
        Emit JSON lines. Defaults to ``LOG_JSON`` (enabled) unless set to
        ``"0"``/``"false"``.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    if json_logs is None:
        json_logs = os.getenv("LOG_JSON", "1").lower() not in {"0", "false", "no"}

    root = logging.getLogger()
    root.setLevel(resolved_level)

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root.handlers.clear()
    root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (call :func:`configure_logging` first)."""
    return logging.getLogger(name)


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
