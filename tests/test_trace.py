"""Durable trace writing (ftw_plan.md §5).

Traces are written directly by the process that owns them, never through
the lossy PUB/SUB tap — a slow or absent subscriber must never lose an
episodic record.
"""

import datetime
import json
import threading

from ftw.protocol import EventEnvelope, EventPayload
from ftw.trace import TraceWriter


def event(trace_id: str, topic: str = "event.tool.shell.exit") -> EventEnvelope:
    return EventEnvelope(
        source="worker.tool.shell",
        target="events",
        trace_id=trace_id,
        payload=EventPayload(topic=topic, data={"code": 0}),
    )


class TestTraceWriter:
    def test_write_creates_dated_jsonl_file(self, tmp_path):
        writer = TraceWriter(tmp_path)
        when = datetime.date(2026, 9, 23)

        path = writer.write(event("trace-1"), when=when)

        assert path == tmp_path / "2026-09-23" / "trace-1.jsonl"
        assert path.exists()

    def test_written_line_round_trips_as_the_envelope(self, tmp_path):
        writer = TraceWriter(tmp_path)
        env = event("trace-1")

        path = writer.write(env, when=datetime.date(2026, 9, 23))

        line = path.read_text().strip()
        from ftw.protocol import parse_envelope

        assert parse_envelope(line) == env

    def test_multiple_writes_append_rather_than_overwrite(self, tmp_path):
        writer = TraceWriter(tmp_path)
        when = datetime.date(2026, 9, 23)
        writer.write(event("trace-1", "event.a"), when=when)
        writer.write(event("trace-1", "event.b"), when=when)

        lines = (tmp_path / "2026-09-23" / "trace-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_different_trace_ids_get_separate_files(self, tmp_path):
        writer = TraceWriter(tmp_path)
        when = datetime.date(2026, 9, 23)
        writer.write(event("trace-1"), when=when)
        writer.write(event("trace-2"), when=when)

        assert (tmp_path / "2026-09-23" / "trace-1.jsonl").exists()
        assert (tmp_path / "2026-09-23" / "trace-2.jsonl").exists()

    def test_defaults_to_todays_date(self, tmp_path):
        writer = TraceWriter(tmp_path)
        path = writer.write(event("trace-1"))
        assert path.parent.name == datetime.datetime.now().astimezone().date().isoformat()

    def test_concurrent_writes_do_not_interleave_or_corrupt_lines(self, tmp_path):
        writer = TraceWriter(tmp_path)
        when = datetime.date(2026, 9, 23)

        def writer_thread(n: int):
            for i in range(25):
                writer.write(event("trace-1", topic=f"event.thread{n}.{i}"), when=when)

        threads = [threading.Thread(target=writer_thread, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = (tmp_path / "2026-09-23" / "trace-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 100
        for line in lines:
            json.loads(line)  # every line must be independently valid JSON
