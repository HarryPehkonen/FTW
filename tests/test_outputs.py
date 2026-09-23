"""Out-of-band tool output store (ftw_plan.md §3.3 "Tool Output Handles").

Raw tool output never enters the prompt wholesale: it's written to
$FTW_HOME/outputs/<trace_id>/<id>, the Turn Horizon gets a bounded excerpt
plus a handle, and the model reads more via read_output/grep_output.
"""

import pytest

from ftw.outputs import OutputNotFound, OutputStore, build_excerpt


class TestPathTraversal:
    """output_id (and trace_id) can reach here from model tool-call
    arguments (read_output/grep_output, agent_loop.py's local tools,
    never pass through the interceptor). An absolute path or a `..`
    segment must never let a read escape the store root — Path("/root")
    / "/etc/hostname" silently evaluates to "/etc/hostname" in Python,
    so naive concatenation is a real path-traversal hole, not a
    theoretical one."""

    def test_absolute_output_id_cannot_read_outside_the_root(self, tmp_path):
        secret = tmp_path.parent / "secret-outside-root.txt"
        secret.write_text("top secret")
        store = OutputStore(tmp_path / "store")

        with pytest.raises(OutputNotFound):
            store.read("trace-1", str(secret))

    def test_dotdot_output_id_cannot_escape_the_root(self, tmp_path):
        # store root is tmp_path/store; two ".." segments from
        # <root>/trace-1/ reach tmp_path itself, where this file lives.
        # POSIX only resolves ".." through a directory that actually
        # exists, so trace-1/ has to be real first (an ordinary save()
        # does that) for this to be a meaningful test of the traversal
        # itself rather than an accidental "no such directory".
        secret = tmp_path / "secret-outside-store.txt"
        secret.write_text("top secret")
        store = OutputStore(tmp_path / "store")
        store.save("trace-1", "ordinary content")

        with pytest.raises(OutputNotFound):
            store.read("trace-1", "../../secret-outside-store.txt")

    def test_absolute_trace_id_cannot_read_outside_the_root(self, tmp_path):
        secret = tmp_path.parent / "secret2.txt"
        secret.write_text("top secret")
        store = OutputStore(tmp_path / "store")

        with pytest.raises(OutputNotFound):
            store.read(str(tmp_path.parent), "secret2.txt")

    def test_grep_is_also_protected(self, tmp_path):
        secret = tmp_path.parent / "secret3.txt"
        secret.write_text("password=hunter2")
        store = OutputStore(tmp_path / "store")

        with pytest.raises(OutputNotFound):
            store.grep("trace-1", str(secret), "password")

    def test_ordinary_generated_ids_still_work(self, tmp_path):
        store = OutputStore(tmp_path / "store")
        output_id = store.save("trace-1", "hello")
        assert store.read("trace-1", output_id) == "hello"


class TestSaveAndRead:
    def test_round_trips_full_content(self, tmp_path):
        store = OutputStore(tmp_path)
        output_id = store.save("trace-1", "line1\nline2\nline3\n")

        assert store.read("trace-1", output_id) == "line1\nline2\nline3\n"

    def test_ids_are_unique_per_save(self, tmp_path):
        store = OutputStore(tmp_path)
        a = store.save("trace-1", "content a")
        b = store.save("trace-1", "content b")
        assert a != b

    def test_different_traces_do_not_collide(self, tmp_path):
        store = OutputStore(tmp_path)
        id1 = store.save("trace-1", "from trace 1")
        id2 = store.save("trace-2", "from trace 2", output_id=id1)

        assert store.read("trace-1", id1) == "from trace 1"
        assert store.read("trace-2", id2) == "from trace 2"

    def test_reading_missing_output_raises(self, tmp_path):
        store = OutputStore(tmp_path)
        with pytest.raises(OutputNotFound):
            store.read("trace-1", "does-not-exist")


class TestLineSlicing:
    def test_read_start_end_selects_line_range(self, tmp_path):
        store = OutputStore(tmp_path)
        output_id = store.save("trace-1", "\n".join(f"line{i}" for i in range(10)))

        assert store.read("trace-1", output_id, start=2, end=5) == "line2\nline3\nline4"


class TestGrep:
    def test_grep_returns_matching_lines(self, tmp_path):
        store = OutputStore(tmp_path)
        output_id = store.save(
            "trace-1",
            "CMake Error: missing OpenSSL\nconfiguring...\nCMake Error: missing zlib\ndone",
        )

        matches = store.grep("trace-1", output_id, r"CMake Error")

        assert matches == ["CMake Error: missing OpenSSL", "CMake Error: missing zlib"]

    def test_grep_respects_max_matches(self, tmp_path):
        store = OutputStore(tmp_path)
        output_id = store.save("trace-1", "\n".join("hit" for _ in range(20)))

        assert len(store.grep("trace-1", output_id, "hit", max_matches=3)) == 3

    def test_grep_on_missing_output_raises(self, tmp_path):
        store = OutputStore(tmp_path)
        with pytest.raises(OutputNotFound):
            store.grep("trace-1", "nope", "pattern")


class TestBuildExcerpt:
    def test_short_content_is_returned_unchanged(self):
        content = "line1\nline2\nline3"
        assert build_excerpt(content, head_lines=5, tail_lines=5) == content

    def test_long_content_is_truncated_with_marker(self):
        content = "\n".join(f"line{i}" for i in range(100))

        excerpt = build_excerpt(content, head_lines=3, tail_lines=3)

        lines = excerpt.splitlines()
        assert lines[:3] == ["line0", "line1", "line2"]
        assert lines[-3:] == ["line97", "line98", "line99"]
        assert "94 lines omitted" in excerpt
