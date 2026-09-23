"""Runtime directory resolution and address formatting (ftw_plan.md §4)."""

import stat

from ftw.runtime import inproc_address, ipc_address, runtime_dir


class TestInprocAddress:
    def test_formats_inproc_url(self):
        assert inproc_address("worker.tool.shell") == "inproc://worker.tool.shell"


class TestRuntimeDir:
    def test_uses_xdg_runtime_dir_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        runtime_dir.cache_clear()
        d = runtime_dir()
        assert d == tmp_path / "ftw"
        runtime_dir.cache_clear()

    def test_falls_back_to_ftw_home_when_xdg_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setenv("FTW_HOME", str(tmp_path / "ftw_home"))
        runtime_dir.cache_clear()
        d = runtime_dir()
        assert d == tmp_path / "ftw_home" / "run"
        runtime_dir.cache_clear()

    def test_creates_dir_with_0700_permissions(self, isolated_runtime_dir):
        d = runtime_dir()
        assert d.is_dir()
        mode = stat.S_IMODE(d.stat().st_mode)
        assert mode == 0o700

    def test_is_idempotent(self, isolated_runtime_dir):
        d1 = runtime_dir()
        d2 = runtime_dir()
        assert d1 == d2


class TestIpcAddress:
    def test_formats_ipc_url_under_runtime_dir(self, isolated_runtime_dir):
        addr = ipc_address("worker.tool.shell")
        d = runtime_dir()
        assert addr == f"ipc://{d / 'worker.tool.shell.ipc'}"
