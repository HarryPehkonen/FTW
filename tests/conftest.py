import uuid

import pytest


@pytest.fixture
def unique_name() -> str:
    """A unique service name per test, so inproc:// addresses never collide."""
    return f"test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def isolated_runtime_dir(tmp_path, monkeypatch):
    """Point FTW's runtime dir at a throwaway tmp_path for ipc:// tests."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("FTW_HOME", str(tmp_path / "ftw_home"))
    from ftw import runtime

    runtime.runtime_dir.cache_clear()
    yield tmp_path
    runtime.runtime_dir.cache_clear()


@pytest.fixture(autouse=True)
def _no_leaked_env(monkeypatch):
    # Keep provider credentials out of the test environment so no test can
    # accidentally make a live network call.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
