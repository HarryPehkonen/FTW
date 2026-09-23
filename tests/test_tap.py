"""ftw tap: the live, decoupled NNG event tap (ftw_plan.md §5).

Zero overhead when unobserved is bus.py's job (PUB sends don't block);
this is just the print loop on the SUB side.
"""

import io
import threading
import time

from ftw.bus import Publisher
from ftw.observability.tap import format_event, run_tap
from ftw.protocol import EventEnvelope, EventPayload
from ftw.runtime import inproc_address


def event(topic: str, data: dict | None = None) -> EventEnvelope:
    return EventEnvelope(
        source="worker.tool.shell", target="events", payload=EventPayload(topic=topic, data=data or {})
    )


class TestFormatEvent:
    def test_includes_topic_and_data(self):
        line = format_event(event("event.tool.shell.exit", {"code": 0}))
        assert "event.tool.shell.exit" in line
        assert "0" in line


class TestRunTap:
    def test_prints_received_events(self, unique_name):
        addr = inproc_address(unique_name)
        out = io.StringIO()
        stop = threading.Event()

        with Publisher(addr) as pub:
            t = threading.Thread(target=run_tap, args=(addr,), kwargs={"stop_event": stop, "out": out}, daemon=True)
            t.start()
            time.sleep(0.05)  # let the subscription establish
            pub.publish(event("event.tool.shell.exit", {"code": 0}))
            time.sleep(0.1)
            stop.set()
            t.join(timeout=1)

        assert "event.tool.shell.exit" in out.getvalue()

    def test_topic_filter_only_prints_matching_events(self, unique_name):
        addr = inproc_address(unique_name)
        out = io.StringIO()
        stop = threading.Event()

        with Publisher(addr) as pub:
            t = threading.Thread(
                target=run_tap,
                args=(addr,),
                kwargs={"stop_event": stop, "out": out, "topics": ["event.tool."]},
                daemon=True,
            )
            t.start()
            time.sleep(0.05)
            pub.publish(event("event.memory.write"))
            pub.publish(event("event.tool.shell.exit"))
            time.sleep(0.1)
            stop.set()
            t.join(timeout=1)

        assert "event.memory.write" not in out.getvalue()
        assert "event.tool.shell.exit" in out.getvalue()

    def test_stop_event_ends_the_loop_promptly(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()

        t = threading.Thread(target=run_tap, args=(addr,), kwargs={"stop_event": stop, "out": io.StringIO()})
        t.start()
        stop.set()
        t.join(timeout=1)

        assert not t.is_alive()
