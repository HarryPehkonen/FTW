"""Out-of-band tool output store (ftw_plan.md §3.3 "Tool Output Handles").

Raw tool output never enters the prompt wholesale: it's written to
$FTW_HOME/outputs/<trace_id>/<id>, the Turn Horizon gets a bounded excerpt
plus a handle, and the model reads more via read_output/grep_output.
"""

import pytest

from ftw.outputs import OutputNotFound, OutputStore, build_excerpt


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
