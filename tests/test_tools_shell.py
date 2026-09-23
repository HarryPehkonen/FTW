"""ShellToolWorker: the standard tool worker, as an independent NNG service
(ftw_plan.md §2, §3.3 "Tool Output Handles").

Most tests call ``handle()`` directly for speed; one test wires it up over
a real Replier/Requester pair to prove the bus plumbing works end to end.
"""

import sys
import threading

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
    def test_runs_command_and_returns_result(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = worker.handle(call([sys.executable, "-c", "print('hi')"]))

        assert isinstance(reply, ResultEnvelope)
        assert reply.payload.status == "ok"
        assert reply.payload.outputs["exit_code"] == 0
        assert reply.parent_span_id is not None

    def test_full_output_is_readable_via_output_store(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "print('hello from the worker')"])

        reply = worker.handle(c)

        output_id = reply.payload.outputs["output_id"]
        full = store.read(c.trace_id, output_id)
        assert "hello from the worker" in full

    def test_long_output_is_excerpted_but_fully_stored(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        script = "for i in range(200): print(f'line{i}')"
        c = call([sys.executable, "-c", script])

        reply = worker.handle(c)

        excerpt = reply.payload.outputs["excerpt"]
        assert "omitted" in excerpt
        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert full.count("\n") >= 199

    def test_nonzero_exit_is_reported_as_error_status(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = worker.handle(call([sys.executable, "-c", "import sys; sys.exit(3)"]))

        assert reply.payload.status == "error"
        assert reply.payload.outputs["exit_code"] == 3

    def test_stderr_is_captured(self, tmp_path):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "import sys; print('oops', file=sys.stderr)"])

        reply = worker.handle(c)

        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert "oops" in full

    def test_respects_cwd_argument(self, tmp_path):
        workdir = tmp_path / "work"
        workdir.mkdir()
        store = OutputStore(tmp_path / "outputs")
        worker = ShellToolWorker(store)
        c = call([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=str(workdir))

        reply = worker.handle(c)

        full = store.read(c.trace_id, reply.payload.outputs["output_id"])
        assert str(workdir) in full


class TestErrorCases:
    def test_unsupported_action_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = worker.handle(call([], action="not_a_real_action"))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "unsupported_action"

    def test_missing_argv_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        c = CallEnvelope(
            source="repl.master",
            target="worker.tool.shell",
            payload=CallPayload(action="run_command", args={}),
        )
        reply = worker.handle(c)

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "invalid_args"

    def test_command_not_found_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path))
        reply = worker.handle(call(["definitely-not-a-real-command-xyz"]))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "not_found"

    def test_timeout_returns_error_envelope(self, tmp_path):
        worker = ShellToolWorker(OutputStore(tmp_path), timeout_s=0.2)
        reply = worker.handle(call([sys.executable, "-c", "import time; time.sleep(2)"]))

        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "timeout"


class TestOverBus:
    def test_served_over_nng_end_to_end(self, tmp_path, unique_name):
        store = OutputStore(tmp_path)
        worker = ShellToolWorker(store)
        addr = inproc_address(unique_name)
        stop = threading.Event()

        with Replier(addr) as rep:
            t = threading.Thread(target=rep.serve_forever, args=(worker.handle, stop), daemon=True)
            t.start()
            with Requester(addr) as req:
                reply = req.call(call([sys.executable, "-c", "print('over the bus')"]))
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.status == "ok"
            stop.set()
