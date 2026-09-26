"""The file-edit tool worker (ftw_plan.md §2, §3.3): literal exact-match
string replacement, as an independent NNG service — the same shape as the
standard shell/process-exec worker (``tools/shell.py``).

Exact-string (not regex) matching sidesteps the shell-quoting pain of doing
this via ``run_command`` + sed/perl, which is where a model is most likely
to silently corrupt a file today. ``old_string`` must match the file's
*current* on-disk content exactly once — 0 or >1 matches is refused, which
both catches "edited the wrong occurrence" and doubles as a cheap
staleness guard: a caller working from a stale reading of the file will
usually fail the uniqueness check rather than silently overwrite unrelated
content. Writes go through a temp-file-then-rename (atomic on POSIX, so a
crash mid-write can never leave a half-written file in place), and a
successful edit's result carries a unified diff so the caller can verify
the change is exactly what was intended.

Confirmation happens upstream, in the Pre-Commit Interceptor pipeline
(§3.4) — by the time a CALL reaches this worker, the human has already
approved it. This worker only edits and reports.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import os
import sys
import tempfile
from pathlib import Path

from ftw.bus import Replier
from ftw.protocol import (
    AnyEnvelope,
    CallEnvelope,
    ErrorEnvelope,
    ErrorPayload,
    ResultEnvelope,
    ResultPayload,
)
from ftw.runtime import ipc_address


class EditToolWorker:
    async def handle(self, envelope: AnyEnvelope) -> ResultEnvelope | ErrorEnvelope:
        if not isinstance(envelope, CallEnvelope):
            return self._error(envelope, "unsupported_action", f"edit worker only accepts CALL, got {envelope.type.value}")
        call = envelope
        if call.payload.action != "edit_file":
            return self._error(call, "unsupported_action", f"unsupported action: {call.payload.action!r}")

        path = call.payload.args.get("path")
        old_string = call.payload.args.get("old_string")
        new_string = call.payload.args.get("new_string")
        if not isinstance(path, str) or not path:
            return self._error(call, "invalid_args", "args.path must be a non-empty string")
        if not isinstance(old_string, str) or not old_string:
            return self._error(call, "invalid_args", "args.old_string must be a non-empty string")
        if not isinstance(new_string, str):
            return self._error(call, "invalid_args", "args.new_string must be a string")
        if old_string == new_string:
            return self._error(call, "no_op", "old_string and new_string are identical")

        try:
            original = await asyncio.to_thread(Path(path).read_text)
        except FileNotFoundError:
            return self._error(call, "not_found", f"no such file: {path!r}")
        except OSError as exc:
            return self._error(call, "io_error", str(exc))

        count = original.count(old_string)
        if count == 0:
            return self._error(call, "no_match", f"old_string not found in {path!r}")
        if count > 1:
            return self._error(
                call, "ambiguous_match", f"old_string matches {count} times in {path!r}, must match exactly once"
            )

        updated = original.replace(old_string, new_string, 1)

        try:
            await asyncio.to_thread(self._atomic_write, path, updated)
        except OSError as exc:
            return self._error(call, "io_error", str(exc))

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
        )

        return ResultEnvelope(
            source=call.target,
            target=call.source,
            parent_span_id=call.span_id,
            payload=ResultPayload(
                status="ok",
                summary=f"edited {path}",
                outputs={"path": path, "diff": diff},
                evidence=[diff],
            ),
        )

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        directory = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _error(envelope: AnyEnvelope, code: str, message: str) -> ErrorEnvelope:
        return ErrorEnvelope(
            source=envelope.target,
            target=envelope.source,
            parent_span_id=envelope.span_id,
            payload=ErrorPayload(code=code, message=message),
        )


async def run_worker(
    address: str,
    *,
    stop_event: asyncio.Event | None = None,
    num_workers: int = 4,
) -> None:
    """Blocks, serving CALLs, until ``stop_event`` is set."""
    worker = EditToolWorker()
    with Replier(address, num_workers=num_workers) as rep:
        await rep.serve_forever(worker.handle, stop_event)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTW file-edit tool worker")
    parser.add_argument("--address", default=ipc_address("worker.tool.edit"))
    args = parser.parse_args(argv)
    print(f"ftw edit worker listening on {args.address}", file=sys.stderr)
    try:
        asyncio.run(run_worker(args.address))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
