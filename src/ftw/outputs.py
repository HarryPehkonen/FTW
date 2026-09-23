"""Out-of-band tool output store (ftw_plan.md §3.3 "Tool Output Handles").

Raw tool output never enters the prompt wholesale — a single build log can
dwarf the entire Turn Horizon budget. It's written here instead, keyed by
trace and a handle id; the Turn Horizon gets a bounded excerpt plus that
handle, and the model reads more with ``read_output``/``grep_output``.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


class OutputNotFound(Exception):
    pass


# output_id and trace_id can reach _path() from model tool-call arguments
# (read_output/grep_output are local tools — they never pass through the
# interceptor). Both must be plain identifiers: no path separators, no
# "..", not absolute. Path("/root") / "/etc/passwd" silently evaluates to
# "/etc/passwd" in Python (an absolute right-hand side replaces the whole
# path), and even a relative ".." can walk back out of the root once an
# intermediate directory genuinely exists — this is a real traversal
# hole if unchecked, not a theoretical one.
_SAFE_ID = re.compile(r"\A[A-Za-z0-9._-]+\Z")


class UnsafeIdentifier(ValueError):
    pass


def _check_safe(value: str, *, what: str) -> None:
    if not _SAFE_ID.match(value) or value in (".", "..") or "/" in value or "\\" in value:
        raise UnsafeIdentifier(f"unsafe {what}: {value!r}")


class OutputStore:
    def __init__(self, root: str | Path):
        self._root = Path(root)

    def save(self, trace_id: str, content: str, *, output_id: str | None = None) -> str:
        output_id = output_id or uuid4().hex
        path = self._path(trace_id, output_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return output_id

    def read(self, trace_id: str, output_id: str, start: int = 0, end: int | None = None) -> str:
        path = self._existing_path(trace_id, output_id)
        if start == 0 and end is None:
            return path.read_text()  # exact round trip, including trailing newline
        return "\n".join(path.read_text().splitlines()[start:end])

    def grep(self, trace_id: str, output_id: str, pattern: str, max_matches: int = 50) -> list[str]:
        regex = re.compile(pattern)
        lines = self._lines(trace_id, output_id)
        matches = [line for line in lines if regex.search(line)]
        return matches[:max_matches]

    def _path(self, trace_id: str, output_id: str) -> Path:
        _check_safe(trace_id, what="trace_id")
        _check_safe(output_id, what="output_id")
        return self._root / trace_id / output_id

    def _existing_path(self, trace_id: str, output_id: str) -> Path:
        try:
            path = self._path(trace_id, output_id)
        except UnsafeIdentifier as exc:
            raise OutputNotFound(str(exc)) from exc
        if not path.exists():
            raise OutputNotFound(f"no output {output_id!r} for trace {trace_id!r}")
        return path

    def _lines(self, trace_id: str, output_id: str) -> list[str]:
        return self._existing_path(trace_id, output_id).read_text().splitlines()


def build_excerpt(content: str, *, head_lines: int = 20, tail_lines: int = 20) -> str:
    """A bounded head/tail excerpt for the Turn Horizon. Content that fits
    within ``head_lines + tail_lines`` is returned unchanged."""
    lines = content.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return content

    omitted = len(lines) - head_lines - tail_lines
    marker = f"… {omitted} lines omitted …"
    return "\n".join([*lines[:head_lines], marker, *lines[-tail_lines:]])
