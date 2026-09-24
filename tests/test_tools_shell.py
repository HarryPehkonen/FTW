"""ShellToolWorker: the standard tool worker, as an independent NNG service
(ftw_plan.md §2, §3.3 "Tool Output Handles").

Most tests call ``handle()`` directly for speed; one test wires it up over
a real Replier/Requester pair to prove the bus plumbing works end to end.
handle() is async, backed by asyncio.create_subprocess_exec (real async
subprocess I/O, not a sync-call-in-a-thread bridge) - see tools/shell.py's
handle() docstring comment for why.
"""

import asyncio
import sys
import time

from ftw.bus import Replier, Requester
from ftw.outputs import OutputStore
from ftw.protocol import CallEnvelope, CallPayload, ErrorEnvelope, ResultEnvelope
from ftw.runtime import inproc_address
from ftw.tools.shell import ShellToolWorker


def call(argv, *, action="run_command", trace_id=None, **extra_args):
    kwargs = {}
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    return CallEnvelope(
        source="repl.master",
        target="worker.tool.shell",
        payload=CallPayload(action=action, args={"argv": argv, **extra_args}),
        **kwargs,
    )


class TestSuccessfulCommand:
    async def test_runs_command_and_returns_result(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = await worker.handle(call([sys.executable, "-c", "print('hi')"]))

        assert isinstance(reply, ResultEnvelope)
        assert reply.payload.status == "ok"
        assert reply.payload.outputs["exit_code"] == 0
        assert reply.parent_span_id is not None

    async def test_full_output_is_readable_via_output_store(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "print('hello from the worker')"])

        reply = await worker.handle(c)

        output_id = reply.payload.outputs["output_id"]
        full = store.read(c.trace_id, output_id)
        assert "hello from the worker" in full

    async def test_long_output_is_excerpted_but_fully_stored(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        script = "for i in range(200): print(f'line{i}')"
        c = call([sys.executable, "-c", script])

        reply = await worker.handle(c)

        excerpt = reply.payload.outputs["excerpt"]
        assert "omitted" in excerpt
        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert full.count("\n") >= 199

    async def test_nonzero_exit_is_reported_as_error_status(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = await worker.handle(call([sys.executable, "-c", "import sys; sys.exit(3)"]))

        assert reply.payload.status == "error"
        assert reply.payload.outputs["exit_code"] == 3

    async def test_stderr_is_captured(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "import sys; print('oops', file=sys.stderr)"])

        reply = await worker.handle(c)

        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert "oops" in full

    async def test_respects_cwd_argument(self, tmp_path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        store = OutputStore(tmp_path / "outputs")
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=str(workdir))

        reply = await worker.handle(c)

        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert str(workdir) in full


class TestErrorCases:
    async def test_unsupported_action_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = await worker.handle(call([], action="not_a_real_action"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "unsupported_action"

    async def test_missing_argv_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        c = CallEnvelope(
            source="repl.master",
            target="worker.tool.shell",
            payload=CallPayload(action="run_command", args={}),
        )
        reply = await worker.handle(c)

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "invalid_args"

    async def test_command_not_found_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = await worker.handle(call(["definitely-not-a-real-command-xyz"]))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "not_found"

    async def test_timeout_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path), timeout_s=0.2)
        reply = await worker.handle(call([sys.executable, "-c", "import time; time.sleep(2)"]))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "timeout"


class TestConcurrency:
    async def test_two_concurrent_commands_run_in_parallel_not_serially(self, tmp_path):
        """The sharpest real-worker-level guard for the concurrency
        invariant: asyncio.create_subprocess_exec must not block the event
        loop while a command runs, or two concurrent run_command calls
        would serialize despite looking concurrent at the call site."""
        worker = ShellToolWorker(OutputStore(tmp_path))
        script = "import time; time.sleep(0.2)"

        t0 = time.monotonic()
        results = await asyncio.gather(
            worker.handle(call([sys.executable, "-c", script])),
            worker.handle(call([sys.executable, "-c", script])),
        )
        elapsed = time.monotonic() - t0

        assert all(r.payload.status == "ok" for r in results)
        # Sequential would take >= 0.4s; concurrent should finish well under that.
        assert elapsed < 0.4


class TestOverBus:
    async def test_served_over_nng_end_to_end(self, tmp_path, unique_name):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        with Replier(addr) as rep:
            task = asyncio.ensure_future(rep.serve_forever(worker.handle, stop))
            with Requester(addr) as req:
                reply = await req.call(call([sys.executable, "-c", "print('over the bus')"]))
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.status == "ok"
            stop.set()
        await task
