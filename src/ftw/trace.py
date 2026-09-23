"""Durable trace logging (ftw_plan.md §5).

Written directly by the process that owns the trace — never through the
PUB/SUB tap, which is lossy by design (see bus.py). Live viewing (``ftw
tap``) and durable history are two different consumers of the same EVENT
envelopes, fed by two different paths.
"""

from __future__ import annotations

import datetime
import threading
from pathlib import Path

from ftw.protocol import EventEnvelope, dump_envelope


class TraceWriter:
    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._lock = threading.Lock()

    def write(self, envelope: EventEnvelope, *, when: datetime.date | None = None) -> Path:
        day = (when or datetime.date.today()).isoformat()
        path = self._root / day / f"{envelope.trace_id}.jsonl"
        line = dump_envelope(envelope).decode("utf-8") + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        return path
