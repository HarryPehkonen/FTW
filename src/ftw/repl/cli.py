"""Wires ReplSession to a real terminal, a real shell worker subprocess,
and real trace/event output (ftw_plan.md §7 Phase 1 deliverable).

``build_repl_session`` does the actual wiring and is what tests call —
worker supervision and a real ipc:// socket, but still no network and no
live model (a scripted MockModelProvider can be passed straight through).
``main`` is the thin ``ftw`` / ``ftw tap`` entry point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from ftw.agent_loop import AgentLoop
from ftw.bus import Publisher, Requester
from ftw.config import build_provider, load_config
from ftw.frames import FrameTree, make_llm_summarizer
from ftw.intercept import ConfirmShellCommands, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.providers import IModelProvider
from ftw.repl.session import InputFn, ReplSession, make_confirm
from ftw.runtime import ipc_address
from ftw.skills.registry import SkillStore
from ftw.trace import TraceWriter
from ftw.workbench import ContextWorkbench

DEFAULT_SYSTEM_ANCHOR = "You are FTW, a local-first agent. Be terse and safe."

# Grace period for the shell-worker subprocess to start. NNG's own dial
# retries transparently in the background (see bus.py), so this isn't
# needed for correctness — a slow-starting worker just adds latency to the
# first real command. What it's actually for is catching a worker that
# never comes up at all (bad import, bind failure, ...): rather than sleep
# through this window blind, _wait_for_worker_or_crash polls the process
# and fails fast and loud the moment it exits, instead of leaving the
# caller to hit a bare deadline_exceeded on its first real command. A full
# readiness handshake over a control channel (ftw_plan.md §4 "Service
# Discovery & Supervision") would let this return early on success too;
# that's future work.
DEFAULT_WORKER_STARTUP_GRACE_S = 0.5
_WORKER_POLL_INTERVAL_S = 0.02


class WorkerStartupError(Exception):
    pass


def spawn_shell_worker(address: str, output_root: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ftw.tools.shell", "--address", address, "--output-root", str(output_root)],
    )


def _wait_for_worker_or_crash(proc: subprocess.Popen, *, grace_s: float) -> None:
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise WorkerStartupError(
                f"shell worker exited during startup (exit code {proc.returncode}); see its output above"
            )
        time.sleep(_WORKER_POLL_INTERVAL_S)


def _enable_readline() -> bool:
    """Enables arrow-key history and line editing on the real interactive
    prompt — stdlib on POSIX, not present on Windows, best-effort either
    way since it's a pure usability nicety."""
    try:
        import readline  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class ReplHandle:
    session: ReplSession
    _worker_proc: subprocess.Popen
    _requester: Requester
    _publisher: Publisher

    def close(self) -> None:
        self._requester.close()
        self._publisher.close()
        self._worker_proc.terminate()
        try:
            self._worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._worker_proc.kill()
            self._worker_proc.wait(timeout=5)


def build_repl_session(
    *,
    ftw_home: Path,
    provider: IModelProvider | None = None,
    config_path: Path | None = None,
    tier: str = "fast",
    input_fn: InputFn = input,
    output: TextIO = sys.stdout,
    system_anchor: str = DEFAULT_SYSTEM_ANCHOR,
    skills_dir: Path | None = None,
    worker_address: str | None = None,
    worker_startup_grace_s: float = DEFAULT_WORKER_STARTUP_GRACE_S,
) -> ReplHandle:
    ftw_home = Path(ftw_home)
    output_root = ftw_home / "outputs"
    traces_root = ftw_home / "traces"
    output_root.mkdir(parents=True, exist_ok=True)

    if provider is None:
        config = load_config(config_path or Path("ftw.toml"))
        provider = build_provider(config, tier)

    worker_address = worker_address or ipc_address(f"worker.tool.shell.{uuid4().hex[:8]}")
    worker_proc = spawn_shell_worker(worker_address, output_root)
    _wait_for_worker_or_crash(worker_proc, grace_s=worker_startup_grace_s)
    requester = Requester(worker_address)

    trace_writer = TraceWriter(traces_root)
    publisher = Publisher(ipc_address(f"events.{uuid4().hex[:8]}"))

    def on_event(evt):
        trace_writer.write(evt)
        publisher.publish(evt)

    workbench = ContextWorkbench(system_anchor=system_anchor)
    skill_store = SkillStore(skills_dir or (ftw_home / "skills"))
    frame_tree = FrameTree(workbench, skill_store, summarizer=make_llm_summarizer(provider))

    loop = AgentLoop(
        workbench=workbench,
        provider=provider,
        output_store=OutputStore(output_root),
        dispatch=requester.call,
        interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
        confirm=make_confirm(input_fn, output),
        frame_tree=frame_tree,
        on_event=on_event,
    )
    session = ReplSession(agent_loop=loop, input_fn=input_fn, output=output)

    return ReplHandle(session=session, _worker_proc=worker_proc, _requester=requester, _publisher=publisher)


def _run_repl(args: argparse.Namespace) -> None:
    handle = build_repl_session(
        ftw_home=Path(args.ftw_home),
        config_path=Path(args.config),
        tier=args.tier,
        skills_dir=Path(args.skills_dir) if args.skills_dir else None,
    )
    try:
        handle.session.run()
    finally:
        handle.close()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    if argv[:1] == ["tap"]:
        from ftw.observability.tap import main as tap_main

        tap_main(argv[1:])
        return

    if argv[:1] == ["shell-worker"]:
        from ftw.tools.shell import main as shell_worker_main

        shell_worker_main(argv[1:])
        return

    parser = argparse.ArgumentParser(
        prog="ftw",
        description="FTW REPL",
        epilog="Other subcommands: 'ftw tap' (stream live events), 'ftw shell-worker' (run the shell worker standalone).",
    )
    parser.add_argument("--config", default="ftw.toml")
    parser.add_argument("--tier", default="fast")
    parser.add_argument("--ftw-home", default=str(Path.home() / ".ftw"))
    parser.add_argument("--skills-dir", default=None, help="defaults to <ftw-home>/skills")
    _enable_readline()
    _run_repl(parser.parse_args(argv))


if __name__ == "__main__":
    main()
