"""Structured JSON access log for anthropic-bridge.

One JSON object per line, written to:
  - $ANTHROPIC_BRIDGE_LOG_FILE (default: logs/anthropic_bridge.log)
  - stdout (so systemd journal captures the same event)

Best-effort and non-blocking: a failed write never breaks the request path.

Producers:
  - server.py HTTP middleware emits request/response access events
  - providers/openai/client.py emits upstream call start/done business events

Consumers:
  - /home/ubuntu/monitor/app.py get_anthropic_bridge_summary() parses this
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_LOG_FILE_ENV = "ANTHROPIC_BRIDGE_LOG_FILE"
_DEFAULT_LOG_PATH = "logs/anthropic_bridge.log"

_lock = threading.Lock()
_file_handle = None
_file_path: Path | None = None


def _resolve_path() -> Path:
    return Path(os.environ.get(_LOG_FILE_ENV) or _DEFAULT_LOG_PATH)


def _ensure_handle():
    """Lazy-open the log file. Best-effort; returns None on failure."""
    global _file_handle, _file_path
    if _file_handle is not None:
        return _file_handle
    with _lock:
        if _file_handle is not None:
            return _file_handle
        try:
            _file_path = _resolve_path()
            _file_path.parent.mkdir(parents=True, exist_ok=True)
            _file_handle = open(_file_path, "a", encoding="utf-8", buffering=1)
        except Exception as e:
            print(
                f'{{"ts":"{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}",'
                f'"level":"error","action":"access_log_open_failed","msg":{json.dumps(str(e))}}}',
                file=sys.stderr,
                flush=True,
            )
            return None
    return _file_handle


def log_event(event: dict[str, Any]) -> None:
    """Emit one structured event. Never raises."""
    event.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback to repr for anything not JSON-serializable
        event = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)) for k, v in event.items()}
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))

    handle = _ensure_handle()
    if handle is not None:
        try:
            handle.write(line + "\n")
        except Exception:
            # Closed / disk full / ENOSPC — swallow
            pass

    # Also emit to stdout so journalctl catches it in case the file is unwritable
    print(line, flush=True)
