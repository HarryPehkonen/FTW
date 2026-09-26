"""EditToolWorker: literal exact-match file editing, as an independent NNG
service (ftw_plan.md §2, §3.3), the same shape as the shell worker
(tools/shell.py).

Exact-string (not regex) replacement sidesteps the shell-quoting pain of
doing this via run_command + sed/perl; a uniqueness check on old_string
(0 or >1 matches is refused) both catches "edited the wrong occurrence"
and doubles as a cheap staleness guard, since old_string has to match the
file's *current* on-disk content, not a stale reading of it. Writes are
temp-file-then-rename (atomic on POSIX), and a successful edit returns a
unified diff so the caller can verify exactly what changed.
"""

import asyncio

from ftw.bus import Replier, Requester
from ftw.protocol import CallEnvelope, CallPayload, ErrorEnvelope, ResultEnvelope
from ftw.runtime import inproc_address
from ftw.tools.edit import EditToolWorker


def call(*, path, old_string, new_string, action="edit_file", trace_id=None, **extra_args):
    kwargs = {}
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    return CallEnvelope(
        source="repl.master",
        target="worker.tool.edit",
        payload=CallPayload(
            action=action,
            args={"path": path, "old_string": old_string, "new_string": new_string, **extra_args},
        ),
        **kwargs,
    )


class TestSuccessfulEdit:
    async def test_replaces_the_unique_match_and_returns_ok(self, tmp_path):
        target = tmp_path / "greeting.txt"
        target.write_text("hello world\ngoodbye world\n")
        worker = EditToolWorker()

        reply = await worker.handle(call(path=str(target), old_string="hello world", new_string="hi world"))

        assert isinstance(reply, ResultEnvelope)
        assert reply.payload.status == "ok"
        assert target.read_text() == "hi world\ngoodbye world\n"
        assert reply.parent_span_id is not None

    async def test_result_includes_a_unified_diff(self, tmp_path):
        target = tmp_path / "greeting.txt"
        target.write_text("hello world\n")
        worker = EditToolWorker()

        reply = await worker.handle(call(path=str(target), old_string="hello", new_string="hi"))

        diff = reply.payload.outputs["diff"]
        assert "-hello world" in diff
        assert "+hi world" in diff

    async def test_edit_is_atomic_the_original_survives_if_something_reads_mid_write(self, tmp_path):
        # Not a true concurrency test (that would need to interrupt the
        # write mid-flight); this documents intent and is covered more
        # directly by test_writes_via_temp_file_then_rename below.
        target = tmp_path / "f.txt"
        target.write_text("a\n")
        worker = EditToolWorker()

        await worker.handle(call(path=str(target), old_string="a", new_string="b"))

        assert target.read_text() == "b\n"

    async def test_writes_via_temp_file_then_rename_no_stray_temp_files_left_behind(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("a\n")
        worker = EditToolWorker()

        await worker.handle(call(path=str(target), old_string="a", new_string="b"))

        assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]

    async def test_only_the_single_matched_occurrence_is_replaced(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("line one\nunique line\nline three\n")
        worker = EditToolWorker()

        await worker.handle(call(path=str(target), old_string="unique line", new_string="replaced line"))

        assert target.read_text() == "line one\nreplaced line\nline three\n"


class TestUniquenessCheck:
    async def test_zero_matches_returns_no_match_error_and_leaves_file_untouched(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("hello world\n")
        worker = EditToolWorker()

        reply = await worker.handle(call(path=str(target), old_string="not present", new_string="x"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "no_match"
        assert target.read_text() == "hello world\n"

    async def test_multiple_matches_returns_ambiguous_match_error_and_leaves_file_untouched(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("dup\ndup\n")
        worker = EditToolWorker()

        reply = await worker.handle(call(path=str(target), old_string="dup", new_string="single"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "ambiguous_match"
        assert target.read_text() == "dup\ndup\n"


class TestErrorCases:
    async def test_unsupported_action_returns_error_envelope(self, tmp_path):
        worker = EditToolWorker()
        reply = await worker.handle(call(path=str(tmp_path / "f.txt"), old_string="a", new_string="b", action="not_a_real_action"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "unsupported_action"

    async def test_missing_path_returns_invalid_args(self, tmp_path):
        worker = EditToolWorker()
        c = CallEnvelope(
            source="repl.master",
            target="worker.tool.edit",
            payload=CallPayload(action="edit_file", args={"old_string": "a", "new_string": "b"}),
        )

        reply = await worker.handle(c)

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "invalid_args"

    async def test_empty_old_string_returns_invalid_args(self, tmp_path):
        worker = EditToolWorker()
        reply = await worker.handle(call(path=str(tmp_path / "f.txt"), old_string="", new_string="b"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "invalid_args"

    async def test_identical_old_and_new_string_returns_no_op(self, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("same\n")
        worker = EditToolWorker()

        reply = await worker.handle(call(path=str(target), old_string="same", new_string="same"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "no_op"

    async def test_file_not_found_returns_not_found_error(self, tmp_path):
        worker = EditToolWorker()
        reply = await worker.handle(call(path=str(tmp_path / "does_not_exist.txt"), old_string="a", new_string="b"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "not_found"


class TestOverBus:
    async def test_served_over_nng_end_to_end(self, tmp_path, unique_name):
        target = tmp_path / "f.txt"
        target.write_text("hello\n")
        worker = EditToolWorker()
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        with Replier(addr) as rep:
            task = asyncio.ensure_future(rep.serve_forever(worker.handle, stop))
            with Requester(addr) as req:
                reply = await req.call(call(path=str(target), old_string="hello", new_string="goodbye"))
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.status == "ok"
            stop.set()
        await task

        assert target.read_text() == "goodbye\n"
