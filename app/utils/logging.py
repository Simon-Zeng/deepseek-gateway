"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
import json
from typing import Any

from app.config import get_settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Add request_id if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "model"):
            log_entry["model"] = record.model
        if hasattr(record, "target_model"):
            log_entry["target_model"] = record.target_model
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging():
    """Configure application logging based on settings."""
    settings = get_settings()
    log_level = getattr(logging, settings.gateway.log_level.upper(), logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Choose format based on config
    log_format = "json"  # Default to JSON
    # Check gateway.yaml for format setting
    try:
        import yaml
        from pathlib import Path
        gateway_yaml = Path(settings.gateway.gateway_config_path)
        if gateway_yaml.exists():
            with open(gateway_yaml, "r") as f:
                config = yaml.safe_load(f) or {}
            log_format = config.get("logging", {}).get("format", "json")
    except Exception:
        pass

    if log_format == "json":
        handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # Quiet down noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
