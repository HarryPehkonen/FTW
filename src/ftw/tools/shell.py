"""The standard shell/process-exec tool worker (ftw_plan.md §2, §3.3).

Runs as an independent NNG service. Full stdout/stderr is written to the
out-of-band output store (never into the reply envelope itself); the reply
carries a bounded excerpt plus the handle the model uses to read more.

Confirmation happens upstream, in the Pre-Commit Interceptor pipeline
(§3.4) — by the time a CALL reaches this worker, the human has already
approved it. This worker only executes and reports.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from ftw.bus import Replier
from ftw.outputs import OutputStore, build_excerpt
from ftw.protocol import CallEnvelope, ErrorEnvelope, ErrorPayload, ResultEnvelope, ResultPayload
from ftw.runtime import ipc_address


class ShellToolWorker:
    def __init__(self, output_store: OutputStore, *, timeout_s: float = 60.0, cwd: str | None = None):
        self._outputs = output_store
        self._default_timeout_s = timeout_s
        self._default_cwd = cwd

    async def handle(self, call: CallEnvelope) -> ResultEnvelope | ErrorEnvelope:
        if call.payload.action != "run_command":
            return self._error(call, "unsupported_action", f"unsupported action: {call.payload.action!r}")

        argv = call.payload.args.get("argv")
        if not isinstance(argv, list) or not argv:
            return self._error(call, "invalid_args", "args.argv must be a non-empty list of strings")

        cwd = call.payload.args.get("cwd", self._default_cwd)
        timeout_s = call.payload.args.get("timeout_s", self._default_timeout_s)

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return self._error(call, "not_found", str(exc))
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            # mirrors subprocess.run(timeout=...)'s own timeout-then-kill
            # behavior, just built from the async subprocess API — real
            # async I/O (not a to_thread bridge) so a hung command is
            # cancellable mid-flight from the caller's side too, not just
            # abandoned in the background.
            proc.kill()
            await proc.wait()
            return self._error(call, "timeout", f"command timed out after {timeout_s}s: {argv!r}")
        wall_time_ms = int((time.monotonic() - start) * 1000)

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        combined = stdout
        if stderr:
            sep = "\n" if combined and not combined.endswith("\n") else ""
            combined += f"{sep}--- stderr ---\n{stderr}"

        output_id = self._outputs.save(call.trace_id, combined)
        excerpt = build_excerpt(combined)

        return ResultEnvelope(
            source=call.target,
            target=call.source,
            parent_span_id=call.span_id,
            payload=ResultPayload(
                status="ok" if proc.returncode == 0 else "error",
                summary=f"exit {proc.returncode}: {' '.join(argv)}",
                outputs={"exit_code": proc.returncode, "output_id": output_id, "excerpt": excerpt},
                evidence=[f"exit code {proc.returncode}"],
                cost={"wall_time_ms": wall_time_ms},
            ),
        )

    @staticmethod
    def _error(call: CallEnvelope, code: str, message: str) -> ErrorEnvelope:
        return ErrorEnvelope(
            source=call.target,
            target=call.source,
            parent_span_id=call.span_id,
            payload=ErrorPayload(code=code, message=message),
        )


async def run_worker(
    address: str,
    output_root: str,
    *,
    stop_event: asyncio.Event | None = None,
    num_workers: int = 4,
) -> None:
    """Blocks, serving CALLs, until ``stop_event`` is set."""
    worker = ShellToolWorker(OutputStore(output_root))
    with Replier(address, num_workers=num_workers) as rep:
        await rep.serve_forever(worker.handle, stop_event)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTW shell tool worker")
    parser.add_argument("--address", default=ipc_address("worker.tool.shell"))
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    print(f"ftw shell worker listening on {args.address}", file=sys.stderr)
    try:
        asyncio.run(run_worker(args.address, args.output_root))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
