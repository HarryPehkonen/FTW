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
from ftw.intercept import ConfirmShellCommands, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.providers import IModelProvider
from ftw.repl.session import InputFn, ReplSession, make_confirm
from ftw.runtime import ipc_address
from ftw.trace import TraceWriter
from ftw.workbench import ContextWorkbench

DEFAULT_SYSTEM_ANCHOR = "You are FTW, a local-first agent. Be terse and safe."

# Pragmatic Phase 1 startup grace: enough for a `python -m` subprocess to
# import and bind its socket before the first real call. A proper
# readiness handshake (rather than a fixed sleep) is future work — see
# ftw_plan.md §4 "Service Discovery & Supervision".
DEFAULT_WORKER_STARTUP_GRACE_S = 0.5


def spawn_shell_worker(address: str, output_root: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "ftw.tools.shell", "--address", address, "--output-root", str(output_root)],
    )


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
    time.sleep(worker_startup_grace_s)
    requester = Requester(worker_address)

    trace_writer = TraceWriter(traces_root)
    publisher = Publisher(ipc_address(f"events.{uuid4().hex[:8]}"))

    def on_event(evt):
        trace_writer.write(evt)
        publisher.publish(evt)

    loop = AgentLoop(
        workbench=ContextWorkbench(system_anchor=system_anchor),
        provider=provider,
        output_store=OutputStore(output_root),
        dispatch=requester.call,
        interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
        confirm=make_confirm(input_fn, output),
        on_event=on_event,
    )
    session = ReplSession(agent_loop=loop, input_fn=input_fn, output=output)

    return ReplHandle(session=session, _worker_proc=worker_proc, _requester=requester, _publisher=publisher)


def _run_repl(args: argparse.Namespace) -> None:
    handle = build_repl_session(
        ftw_home=Path(args.ftw_home),
        config_path=Path(args.config),
        tier=args.tier,
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
    _run_repl(parser.parse_args(argv))


if __name__ == "__main__":
    main()
