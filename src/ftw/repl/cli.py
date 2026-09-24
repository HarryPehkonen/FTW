"""Wires ReplSession to a real terminal, a real shell worker subprocess,
and real trace/event output (ftw_plan.md §7 Phase 1 deliverable).

``build_repl_session`` does the actual wiring and is what tests call —
worker supervision and a real ipc:// socket, but still no network and no
live model (a scripted MockModelProvider can be passed straight through).
``main`` is the thin ``ftw`` / ``ftw tap`` entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from ftw.agent_loop import AgentLoop
from ftw.bus import DispatchRouter, Publisher, Requester
from ftw.config import build_provider, load_config
from ftw.frames import FrameTree, make_llm_summarizer
from ftw.intercept import ConfirmShellCommands, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.providers import IModelProvider
from ftw.repl.session import InputFn, ReplSession, make_ask_answerer
from ftw.runtime import ipc_address
from ftw.skills.registry import SkillStore
from ftw.skills.runner import control_address as skill_runner_control_address
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

# Logical target names, as recorded in a CallEnvelope's `target` field and
# looked up by DispatchRouter — not the actual dynamic ipc:// address,
# which changes on every run.
SHELL_WORKER_TARGET = "worker.tool.shell"  # matches agent_loop.DEFAULT_TOOL_TARGETS
SKILL_RUNNER_TARGET = "skill.runner"


class WorkerStartupError(Exception):
    pass


def spawn_shell_worker(address: str, output_root: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ftw.tools.shell", "--address", address, "--output-root", str(output_root)],
    )


def spawn_skill_runner_worker(
    address: str,
    *,
    skills_dir: Path,
    output_root: Path,
    shell_worker_address: str,
    config_path: Path,
    default_tier: str,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ftw.skills.runner",
            "--address",
            address,
            "--skills-dir",
            str(skills_dir),
            "--output-root",
            str(output_root),
            "--shell-worker-address",
            shell_worker_address,
            "--config",
            str(config_path),
            "--default-tier",
            default_tier,
        ],
    )


async def _wait_for_worker_or_crash(proc: subprocess.Popen, *, grace_s: float, name: str = "worker") -> None:
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise WorkerStartupError(f"{name} exited during startup (exit code {proc.returncode}); see its output above")
        await asyncio.sleep(_WORKER_POLL_INTERVAL_S)


def _enable_readline() -> bool:
    """Enables arrow-key history and line editing on the real interactive
    prompt — stdlib on POSIX, not present on Windows, best-effort either
    way since it's a pure usability nicety."""
    try:
        import readline  # noqa: F401
    except ImportError:
        return False
    return True


def _stop_worker(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@dataclass
class ReplHandle:
    session: ReplSession
    _shell_worker_proc: subprocess.Popen
    _shell_requester: Requester
    _skill_runner_proc: subprocess.Popen
    _skill_runner_requester: Requester
    _skill_runner_control_requester: Requester
    _publisher: Publisher

    def close(self) -> None:
        self._shell_requester.close()
        self._skill_runner_requester.close()
        self._skill_runner_control_requester.close()
        self._publisher.close()
        _stop_worker(self._shell_worker_proc)
        _stop_worker(self._skill_runner_proc)


async def build_repl_session(
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
    skill_runner_address: str | None = None,
    events_address: str | None = None,
    worker_startup_grace_s: float = DEFAULT_WORKER_STARTUP_GRACE_S,
) -> ReplHandle:
    ftw_home = Path(ftw_home)
    output_root = ftw_home / "outputs"
    traces_root = ftw_home / "traces"
    output_root.mkdir(parents=True, exist_ok=True)

    if provider is None:
        config = load_config(config_path or Path("ftw.toml"))
        provider = build_provider(config, tier)

    resolved_skills_dir = skills_dir or (ftw_home / "skills")
    resolved_config_path = config_path or Path("ftw.toml")

    shell_worker_address = worker_address or ipc_address(f"worker.tool.shell.{uuid4().hex[:8]}")
    shell_worker_proc = spawn_shell_worker(shell_worker_address, output_root)
    await _wait_for_worker_or_crash(shell_worker_proc, grace_s=worker_startup_grace_s, name="shell worker")
    shell_requester = Requester(shell_worker_address)

    skill_runner_address = skill_runner_address or ipc_address(f"skill.runner.{uuid4().hex[:8]}")
    skill_runner_proc = spawn_skill_runner_worker(
        skill_runner_address,
        skills_dir=resolved_skills_dir,
        output_root=output_root,
        shell_worker_address=shell_worker_address,
        config_path=resolved_config_path,
        default_tier=tier,
    )
    await _wait_for_worker_or_crash(skill_runner_proc, grace_s=worker_startup_grace_s, name="skill runner")
    skill_runner_requester = Requester(skill_runner_address)
    skill_runner_control_requester = Requester(skill_runner_control_address(skill_runner_address))

    trace_writer = TraceWriter(traces_root)
    # Fixed, well-known address by default — matches observability/tap.py's
    # own default exactly, on purpose (both call ipc_address("events")),
    # so `uv run ftw tap` with no arguments finds a real running session.
    # A random per-session address is only for tests/multi-session use.
    publisher = Publisher(events_address or ipc_address("events"))

    def on_event(evt):
        trace_writer.write(evt)
        publisher.publish(evt)

    workbench = ContextWorkbench(system_anchor=system_anchor)
    skill_store = SkillStore(resolved_skills_dir)
    frame_tree = FrameTree(workbench, skill_store, summarizer=make_llm_summarizer(provider))

    dispatch = DispatchRouter({SHELL_WORKER_TARGET: shell_requester, SKILL_RUNNER_TARGET: skill_runner_requester})

    loop = AgentLoop(
        workbench=workbench,
        provider=provider,
        output_store=OutputStore(output_root),
        dispatch=dispatch,
        interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
        ask_answerer=make_ask_answerer(input_fn, output),
        control_dispatch=skill_runner_control_requester.call,
        frame_tree=frame_tree,
        skill_runner_target=SKILL_RUNNER_TARGET,
        on_event=on_event,
    )
    session = ReplSession(agent_loop=loop, input_fn=input_fn, output=output)

    return ReplHandle(
        session=session,
        _shell_worker_proc=shell_worker_proc,
        _shell_requester=shell_requester,
        _skill_runner_proc=skill_runner_proc,
        _skill_runner_requester=skill_runner_requester,
        _skill_runner_control_requester=skill_runner_control_requester,
        _publisher=publisher,
    )


async def _run_repl(args: argparse.Namespace) -> None:
    handle = await build_repl_session(
        ftw_home=Path(args.ftw_home),
        config_path=Path(args.config),
        tier=args.tier,
        skills_dir=Path(args.skills_dir) if args.skills_dir else None,
    )
    try:
        await handle.session.run()
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

    if argv[:1] == ["skill-runner"]:
        from ftw.skills.runner import main as skill_runner_main

        skill_runner_main(argv[1:])
        return

    parser = argparse.ArgumentParser(
        prog="ftw",
        description="FTW REPL",
        epilog=(
            "Other subcommands: 'ftw tap' (stream live events), 'ftw shell-worker' "
            "(run the shell worker standalone), 'ftw skill-runner' (run the delegated skill worker standalone)."
        ),
    )
    parser.add_argument("--config", default="ftw.toml")
    parser.add_argument("--tier", default="fast")
    parser.add_argument("--ftw-home", default=str(Path.home() / ".ftw"))
    parser.add_argument("--skills-dir", default=None, help="defaults to <ftw-home>/skills")
    _enable_readline()
    asyncio.run(_run_repl(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
