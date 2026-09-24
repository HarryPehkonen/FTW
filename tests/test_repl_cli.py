"""Worker supervision and end-to-end wiring (ftw_plan.md §7 Phase 1
deliverable: "REPL mounts... runs confirmed shell commands over NNG").

This is the one place a real subprocess and a real ipc:// socket are
exercised — everything else in the suite stays over inproc://. Still zero
network calls and zero live LLM tokens: the model side is MockModelProvider.
build_repl_session/_wait_for_worker_or_crash and AgentLoop.run_turn are all
async - see cli.py's and agent_loop.py's module docstrings for why.
"""

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.repl.cli import WorkerStartupError, _enable_readline, _wait_for_worker_or_crash, build_repl_session


class ScriptedInput:
    """Mimics input(): pops the next scripted line, raises EOFError once
    exhausted."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def __call__(self, prompt: str = "") -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def assistant_run_command() -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", name="run_command", arguments={"argv": [sys.executable, "-c", "print('from the real worker')"]})],
        )
    )


def assistant_text(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


class TestEnableReadline:
    def test_returns_true_when_readline_is_importable(self):
        # readline is stdlib on this (Linux) test environment.
        assert _enable_readline() is True

    def test_returns_false_without_raising_when_unavailable(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "readline":
                raise ImportError("no module named readline")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert _enable_readline() is False  # must not raise, e.g. on Windows


class TestWaitForWorkerOrCrash:
    async def test_returns_quietly_when_the_process_stays_alive(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        try:
            await _wait_for_worker_or_crash(proc, grace_s=0.2)  # must not raise
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    async def test_raises_promptly_when_the_process_exits_early(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
        start = time.monotonic()
        try:
            with pytest.raises(WorkerStartupError, match="3"):
                await _wait_for_worker_or_crash(proc, grace_s=5.0)
        finally:
            proc.wait(timeout=5)
        assert time.monotonic() - start < 2.0  # detected the crash, didn't wait out the full grace period


class TestBuildReplSessionWithRealWorker:
    async def test_shell_command_round_trips_through_a_real_subprocess_worker(self, isolated_runtime_dir, tmp_path):
        provider = MockModelProvider([assistant_run_command(), assistant_text("it printed the line")])
        out = io.StringIO()

        handle = await build_repl_session(
            ftw_home=tmp_path,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the one shell command
            output=out,
        )
        try:
            reply = await handle.session.agent_loop.run_turn("run something")
        finally:
            handle.close()

        assert reply == "it printed the line"

    async def test_default_events_address_matches_what_ftw_tap_dials_by_default(self, isolated_runtime_dir, tmp_path):
        """ftw tap's own default (--address, observability/tap.py) has to
        be the same address the REPL actually publishes on with no
        override — otherwise `uv run ftw tap` silently receives nothing
        against a real running session, which it did before this fix."""
        from ftw.bus import Subscriber
        from ftw.runtime import ipc_address

        provider = MockModelProvider([assistant_text("hello")])
        handle = await build_repl_session(
            ftw_home=tmp_path,
            provider=provider,
            input_fn=ScriptedInput([]),
            output=io.StringIO(),
        )
        try:
            with Subscriber(ipc_address("events"), topics=[""]) as sub:  # exactly tap.py's own default
                time.sleep(0.05)  # let the subscription establish
                await handle.session.agent_loop.run_turn("hi")
                received = sub.recv(timeout_ms=2000)
            assert received.payload.topic.startswith("event.")
        finally:
            handle.close()


class TestRealSigintDuringATurn:
    """The actual real-world scenario this whole async migration exists to
    fix: a real SIGINT arriving while a turn is genuinely blocked
    dispatching to a real subprocess worker must interrupt it promptly -
    not leave the REPL wedged until run_command's own 65s deadline
    eventually elapses on its own. Everything below this test class
    already proves the pieces in isolation (bus.py's TestRecvIsInterruptible,
    agent_loop.py's real-task.cancel() test); this is the one place they're
    all proven together, end to end, over a real ipc:// socket to a real
    subprocess."""

    async def test_real_sigint_cancels_a_turn_blocked_on_a_real_subprocess_worker(self, isolated_runtime_dir, tmp_path):
        # Cancellation during dispatch is caught and reported as ordinary
        # tool content (tested at the unit level in
        # test_agent_loop.py::TestCancelledDuringDispatch), so the turn
        # continues to a second model round rather than the whole turn
        # dying with it - hence two scripted responses, matching that same
        # pattern, not one.
        provider = MockModelProvider(
            [
                tool_call_response("run_command", {"argv": [sys.executable, "-c", "import time; time.sleep(30)"]}, "c1"),
                assistant_text("cancelled that for you"),
            ]
        )
        out = io.StringIO()
        handle = await build_repl_session(
            ftw_home=tmp_path,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the shell command
            output=out,
        )
        try:

            def send_sigint_soon():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=send_sigint_soon, daemon=True).start()

            start = time.monotonic()
            reply = await handle.session._run_turn_interruptible("run something long")  # noqa: SLF001 - whitebox: this IS what Ctrl-C drives
            elapsed = time.monotonic() - start
        finally:
            handle.close()

        assert reply == "cancelled that for you"
        assert elapsed < 5.0  # nowhere near run_command's own 65s deadline


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def tool_call_response(name: str, arguments: dict, call_id: str = "c") -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])
    )


def text_response(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


class TestPhase2Deliverable:
    """ftw_plan.md §8 Phase 2: "The model finds and mounts a skill (and a
    nested helper), works multi-turn with shell tools, and unmounts.
    /context shows the footprint back at baseline plus milestones." Proven
    against a real subprocess shell worker and real SKILL.md files on
    disk, across two genuinely separate turns — only the model side is
    mocked (and note that the *same* mocked provider also has to answer
    the milestone summarizer's calls, in the order they actually happen:
    two more completions, child-then-parent, fired from inside the
    unmount_skill tool call itself, before the turn's final answer)."""

    async def test_find_mount_nested_mount_run_shell_then_unmount_restores_baseline(self, isolated_runtime_dir, tmp_path):
        skills_dir = tmp_path / "skills"
        write_skill(skills_dir, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
        write_skill(skills_dir, "toolchain/verify_installed", "toolchain.verify_installed", "Verify a compiler toolchain.")

        provider = MockModelProvider(
            [
                # --- turn 1: find, mount (nested), run a confirmed shell command, answer ---
                tool_call_response("find_skill", {"query": "cmake configuration"}, "c1"),
                tool_call_response("mount_skill", {"name": "cmake.diagnose_configure"}, "c2"),
                tool_call_response("mount_skill", {"name": "toolchain.verify_installed"}, "c3"),
                tool_call_response(
                    "run_command", {"argv": [sys.executable, "-c", "print('gcc found')"]}, "c4"
                ),
                text_response("mounted cmake help and confirmed gcc is on PATH"),
                # --- turn 2: unmount (triggers two summarizer calls, child then parent), answer ---
                tool_call_response("unmount_skill", {"name": "cmake.diagnose_configure"}, "c5"),
                text_response("toolchain.verify_installed: gcc found on PATH."),  # summarizer call, child
                text_response("cmake.diagnose_configure: root cause identified."),  # summarizer call, parent
                text_response("all done"),
            ]
        )
        out = io.StringIO()
        handle = await build_repl_session(
            ftw_home=tmp_path,
            skills_dir=skills_dir,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the one shell command
            output=out,
        )
        workbench = handle.session.agent_loop.workbench

        try:
            first_reply = await handle.session.agent_loop.run_turn("diagnose my cmake failure")
            assert workbench.mounted_skill_tokens > 0  # still mounted between turns
            assert workbench.milestones == []

            second_reply = await handle.session.agent_loop.run_turn("looks good, wrap it up")
        finally:
            handle.close()

        assert first_reply == "mounted cmake help and confirmed gcc is on PATH"
        assert second_reply == "all done"

        assert workbench.mounted_skill_tokens == 0  # the whole subtree was evicted
        assert len(workbench.milestones) == 1  # one combined milestone, child folded into parent
        assert workbench.milestones[0] == "cmake.diagnose_configure: root cause identified. [toolchain.verify_installed: gcc found on PATH.]"
        # Per-message tagging (not per-turn): only the messages actually
        # produced while cmake/toolchain had focus were evicted. Turn 1's
        # opening exchange ("diagnose my cmake failure", finding the
        # skill, deciding to mount it) happened *before* the mount took
        # effect, so it correctly survives — unlike the old per-turn
        # tagging, which tagged (and so destroyed) the whole turn based on
        # whatever was focused only once the turn finished. The mounted
        # work itself (both mount results, the confirmed shell command,
        # the "mounted..." reply) is gone.
        assert len(workbench.turns) == 2
        turn1_contents = [m.content for m in workbench.turns[0]]
        assert turn1_contents[0] == "diagnose my cmake failure"
        assert "mounted cmake help and confirmed gcc is on PATH" not in turn1_contents
        assert not any(isinstance(c, str) and "gcc found" in c for c in turn1_contents)
        # Turn 2 (mount -> unmount all within one reply, the model's
        # natural self-mount pattern) wasn't committed to the workbench
        # until *after* its own unmount already ran, so evict_frame()
        # never had a chance to reach into it — it survives whole. This
        # is expected: eviction only ever acts on already-committed
        # history, never on the turn still being built.
        assert workbench.turns[1][-1].content == "all done"

        snapshot = workbench.snapshot()
        by_name = {z.name: z for z in snapshot.zones}
        assert by_name["Mounted Skill"].tokens == 0
        assert by_name["Milestones"].detail == "1 milestone(s)"


def openai_tool_call_response(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}],
                }
            }
        ]
    }


def openai_text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class _ScriptedOpenAIHandler(BaseHTTPRequestHandler):
    """A minimal local stand-in for an OpenAI-compatible /chat/completions
    endpoint — fully local and scripted, not a real provider, so this
    still respects "no live LLM tokens." It's what lets the skill-runner
    *subprocess* (which resolves a real OpenAICompatibleProvider from
    ftw.toml, since MockModelProvider can't cross a process boundary)
    actually get a deterministic answer. A plain sync HTTP server is fine
    here even though the subprocess's own client is httpx.AsyncClient now
    — HTTP itself doesn't care whether either side is sync or async."""

    responses: list[dict] = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's naming convention
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(self.responses.pop(0)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - silence per-request logging in test output
        pass


def start_mock_model_server(responses: list[dict]) -> HTTPServer:
    handler_cls = type("_Handler", (_ScriptedOpenAIHandler,), {"responses": list(responses)})
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestDelegatedSkillRealSubprocess:
    """ftw_plan.md §8 Phase 3 deliverable: "A mounted skill or the REPL
    delegates a sub-task; caller context grows only by the Result." Proven
    against a real skill-runner subprocess talking to a real (if scripted
    and local) HTTP endpoint for its own model calls — the one boundary
    other Phase 3 tests can't cross, since MockModelProvider can't be
    handed to a separate process."""

    async def test_delegate_skill_across_a_real_subprocess_boundary(self, isolated_runtime_dir, tmp_path):
        skills_dir = tmp_path / "skills"
        write_skill(skills_dir, "toolchain/verify_installed", "toolchain.verify_installed", "Verify a compiler toolchain is installed.")

        server = start_mock_model_server(
            [openai_tool_call_response("submit_result", {"status": "ok", "summary": "gcc 13 is installed and on PATH."})]
        )
        try:
            config_path = tmp_path / "ftw.toml"
            config_path.write_text(
                f"""
                [providers.mock_http]
                kind = "openai_compatible"
                base_url = "http://127.0.0.1:{server.server_port}/v1"

                [tiers.fast]
                provider = "mock_http"
                model = "mock-model"
                """
            )

            main_provider = MockModelProvider(
                [
                    tool_call_response("delegate_skill", {"name": "toolchain.verify_installed", "brief": "check gcc"}, "d1"),
                    text_response("Confirmed: gcc 13 is installed and on PATH."),
                ]
            )
            out = io.StringIO()
            handle = await build_repl_session(
                ftw_home=tmp_path,
                skills_dir=skills_dir,
                config_path=config_path,
                provider=main_provider,
                input_fn=ScriptedInput([]),
                output=out,
            )
            workbench = handle.session.agent_loop.workbench
            try:
                reply = await handle.session.agent_loop.run_turn("is gcc installed?")
            finally:
                handle.close()
        finally:
            server.shutdown()

        assert reply == "Confirmed: gcc 13 is installed and on PATH."
        # the caller's context grew by exactly one tool message: the
        # delegated run's structured summary, nothing about its internals
        tool_messages = [m for m in workbench.turns[0] if m.role == ChatRole.TOOL]
        assert len(tool_messages) == 1
        assert "gcc 13 is installed" in tool_messages[0].content

    async def test_delegated_run_asking_for_confirmation_is_relayed_across_the_real_subprocess_boundary(self, isolated_runtime_dir, tmp_path):
        """The skill-runner subprocess's own interceptor asks for
        confirmation before running a shell command; that ASK crosses back
        to this process, gets answered here (scripted 'y'), and the answer
        crosses back to resume the *same* subprocess run — proving ASK
        passthrough end to end, not just at the unit level."""
        skills_dir = tmp_path / "skills"
        write_skill(skills_dir, "cmake/clean_build", "cmake.clean_build", "Clean stale CMake build artifacts.")

        server = start_mock_model_server(
            [
                openai_tool_call_response("run_command", {"argv": [sys.executable, "-c", "print('removed build/')"]}),
                openai_tool_call_response("submit_result", {"status": "ok", "summary": "Removed stale build artifacts."}),
            ]
        )
        try:
            config_path = tmp_path / "ftw.toml"
            config_path.write_text(
                f"""
                [providers.mock_http]
                kind = "openai_compatible"
                base_url = "http://127.0.0.1:{server.server_port}/v1"

                [tiers.fast]
                provider = "mock_http"
                model = "mock-model"
                """
            )

            main_provider = MockModelProvider(
                [
                    tool_call_response("delegate_skill", {"name": "cmake.clean_build", "brief": "clean the build dir"}, "d1"),
                    text_response("Build directory cleaned."),
                ]
            )
            out = io.StringIO()
            handle = await build_repl_session(
                ftw_home=tmp_path,
                skills_dir=skills_dir,
                config_path=config_path,
                provider=main_provider,
                input_fn=ScriptedInput(["y"]),  # answers the relayed ASK
                output=out,
            )
            try:
                reply = await handle.session.agent_loop.run_turn("clean the build directory")
            finally:
                handle.close()
        finally:
            server.shutdown()

        assert reply == "Build directory cleaned."
        assert "shell commands require confirmation" in out.getvalue()  # the relayed ASK was actually shown here


class TestCancelRealSubprocess:
    """Proves the control channel (ftw_plan.md §4: CANCEL on a separate
    ``<service>.ctl`` socket, "so a busy call channel can't block it") is a
    genuinely separate, working NNG socket in the real skill-runner
    subprocess — not just exercised in-process against the worker object
    directly, the way test_skills_runner.py's TestCancel does."""

    async def test_cancel_over_the_real_control_channel_aborts_a_real_subprocess_run(self, isolated_runtime_dir, tmp_path):
        skills_dir = tmp_path / "skills"
        write_skill(skills_dir, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose CMake.")

        # A provider that's never actually dialed: the cancel is checked
        # before the first model call, so no live endpoint is needed here.
        config_path = tmp_path / "ftw.toml"
        config_path.write_text(
            """
            [providers.stub]
            kind = "openai_compatible"
            base_url = "http://127.0.0.1:1/v1"

            [tiers.fast]
            provider = "stub"
            model = "unused"
            """
        )

        handle = await build_repl_session(
            ftw_home=tmp_path,
            skills_dir=skills_dir,
            config_path=config_path,
            provider=MockModelProvider([]),  # the main REPL loop isn't exercised in this test
            input_fn=ScriptedInput([]),
            output=io.StringIO(),
        )
        try:
            from ftw.protocol import CallEnvelope, CallPayload, CancelEnvelope, CancelPayload, ResultEnvelope

            span_id = "known-span-for-cancel-test"
            cancel = CancelEnvelope(source="repl.master", target="skill.runner", payload=CancelPayload(target_span_id=span_id))
            cancel_reply = await handle._skill_runner_control_requester.call(cancel)  # noqa: SLF001 - whitebox: exercising the real control socket directly
            assert isinstance(cancel_reply, ResultEnvelope)

            call = CallEnvelope(
                source="repl.master",
                target="skill.runner",
                span_id=span_id,
                payload=CallPayload(action="delegate_skill", args={"name": "cmake.diagnose_configure", "brief": "go"}),
            )
            reply = await handle._skill_runner_requester.call(call)  # noqa: SLF001
        finally:
            handle.close()

        assert isinstance(reply, ResultEnvelope)
        assert reply.payload.status == "error"
        assert "cancel" in reply.payload.summary.lower()
