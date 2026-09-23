"""Runtime directory resolution and transport address formatting (ftw_plan.md §4).

Unix domain sockets live under ``$XDG_RUNTIME_DIR/ftw/`` (falling back to
``$FTW_HOME/run/``), never under ``/tmp`` — a shell worker listening on a
world-reachable socket is a local privilege hole, and it also keeps the
socket path well under the ~107-byte ``sun_path`` limit.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def runtime_dir() -> Path:
    """Resolve (and create, mode 0700) the directory FTW's ipc:// sockets live in.

    Cached for the process lifetime; tests that change the environment must
    call ``runtime_dir.cache_clear()`` first (see ``isolated_runtime_dir``
    in tests/conftest.py).
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        base = Path(xdg) / "ftw"
    else:
        ftw_home = os.environ.get("FTW_HOME", str(Path.home() / ".ftw"))
        base = Path(ftw_home) / "run"

    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    return base


def inproc_address(name: str) -> str:
    """The in-process transport address for a service, for tests."""
    return f"inproc://{name}"


def ipc_address(name: str) -> str:
    """The Unix-domain-socket transport address for a service, for runtime use."""
    return f"ipc://{runtime_dir() / f'{name}.ipc'}"
