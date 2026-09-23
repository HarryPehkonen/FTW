"""``ftw tap``: the decoupled, live NNG event stream (ftw_plan.md §5).

Purely a viewer — durable history is trace.py's job, written directly by
the owning process. If this tap isn't running, PUB sends are simply
discarded by NNG; nothing here is required for correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import TextIO

from ftw.bus import DeadlineExceeded, Subscriber
from ftw.protocol import EventEnvelope
from ftw.runtime import ipc_address

DEFAULT_POLL_MS = 100


def format_event(envelope: EventEnvelope) -> str:
    return f"[{envelope.trace_id[:8]}] {envelope.payload.topic} {json.dumps(envelope.payload.data)}"


def run_tap(
    address: str,
    *,
    topics: list[str] | None = None,
    stop_event: threading.Event | None = None,
    out: TextIO = sys.stdout,
    poll_ms: int = DEFAULT_POLL_MS,
) -> None:
    """Blocks, printing one formatted line per event, until ``stop_event``
    is set (or forever if omitted)."""
    with Subscriber(address, topics=topics or [""], recv_timeout_ms=poll_ms) as sub:
        while stop_event is None or not stop_event.is_set():
            try:
                envelope = sub.recv()
            except DeadlineExceeded:
                continue
            print(format_event(envelope), file=out, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stream live FTW events")
    parser.add_argument("--address", default=ipc_address("events"))
    parser.add_argument("--topic", action="append", default=None, help="filter to a topic prefix (repeatable)")
    args = parser.parse_args(argv)
    try:
        run_tap(args.address, topics=args.topic)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
