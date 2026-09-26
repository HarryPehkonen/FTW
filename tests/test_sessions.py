"""SessionStore: persists a ContextWorkbench (and, if given, a FrameTree)
to a JSON file and restores them later (ftw_plan.md §3.6 /save, /load) -
"files are the source of truth" (CLAUDE.md): a saved session is a plain,
inspectable JSON file under $FTW_HOME/sessions, not an opaque blob.

Frame ids are process-lifetime UUIDs, not stable across a restart, so a
mounted frame is serialized by its skill_name instead — FrameTree already
guarantees a skill_name is unique among currently-mounted frames (mount()
raises SkillAlreadyMounted otherwise), which makes it a valid stable key.
load() remounts each saved frame (in saved order, so parents remount
before children) to get fresh, valid ids, then re-tags every restored
turn/pin through the resulting skill_name -> new-id mapping — the same
tagging ContextWorkbench.add_tagged_turn/pin already do for a live mount,
just replayed from a save file instead of live tool calls. A frame that
can't be remounted (skill removed/renamed on disk, budget doesn't fit
this time) is reported in LoadResult.failed_skills rather than raised -
its content is never dropped, just re-tagged to the root session instead
of the frame that no longer exists (see TestFrameFidelity below).
"""

import pytest

from ftw.frames import FrameTree
from ftw.providers import ChatMessage, ChatRole, ToolCall
from ftw.sessions import (
    DEFAULT_SESSION_NAME,
    SessionNotFound,
    SessionStore,
    UnsafeSessionName,
)
from ftw.skills.registry import SkillStore
from ftw.workbench import ContextWorkbench


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.USER, content=text)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.ASSISTANT, content=text)


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    write_skill(d, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
    write_skill(d, "cmake/inspect_cache", "cmake.inspect_cache", "Inspect CMakeCache.txt for a missing variable.")
    write_skill(d, "git/bisect", "git.bisect", "Binary-search history for a regression.")
    return d


def make_tree(skills_dir) -> tuple[ContextWorkbench, FrameTree]:
    wb = ContextWorkbench()
    return wb, FrameTree(wb, SkillStore(skills_dir))


class TestSaveAndLoadRoundTripNoFrameTree:
    """No mounted skills involved at all - the plain conversation case."""

    async def test_turns_round_trip(self, tmp_path):
        wb = ContextWorkbench()
        wb.add_turn([user("hello"), assistant("hi there")])
        store = SessionStore(tmp_path)

        store.save(wb)
        restored = ContextWorkbench()
        result = await store.load(restored)

        assert result.turn_count == 1
        assert restored.turns == [[user("hello"), assistant("hi there")]]
        assert result.mounted_skills == []
        assert result.failed_skills == []

    async def test_milestones_and_scratchpad_round_trip(self, tmp_path):
        wb = ContextWorkbench()
        wb.add_milestone("diagnosed the build failure")
        wb.pin("repo", "/home/user/src/core")
        store = SessionStore(tmp_path)

        store.save(wb)
        restored = ContextWorkbench()
        await store.load(restored)

        assert restored.milestones == ["diagnosed the build failure"]
        assert restored.scratchpad == {"repo": "/home/user/src/core"}

    async def test_tool_calls_round_trip(self, tmp_path):
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="call-1", name="run_command", arguments={"argv": ["echo", "hi"]})],
        )
        wb = ContextWorkbench()
        wb.add_turn([user("run echo"), msg])
        store = SessionStore(tmp_path)

        store.save(wb)
        restored = ContextWorkbench()
        await store.load(restored)

        assert restored.turns[0][1].tool_calls[0].name == "run_command"
        assert restored.turns[0][1].tool_calls[0].arguments == {"argv": ["echo", "hi"]}

    async def test_default_name_is_latest(self, tmp_path):
        wb = ContextWorkbench()
        wb.add_turn([user("hi")])
        store = SessionStore(tmp_path)

        store.save(wb)

        assert (tmp_path / f"{DEFAULT_SESSION_NAME}.json").exists()

    async def test_named_session_does_not_clobber_a_differently_named_one(self, tmp_path):
        store = SessionStore(tmp_path)
        wb_a = ContextWorkbench()
        wb_a.add_turn([user("session a")])
        wb_b = ContextWorkbench()
        wb_b.add_turn([user("session b")])

        store.save(wb_a, name="a")
        store.save(wb_b, name="b")

        restored = ContextWorkbench()
        await store.load(restored, name="a")
        assert restored.turns == [[user("session a")]]


class TestFrameFidelity:
    """The point of the whole exercise: a skill that's still mounted at
    save time comes back mounted, with its history correctly re-tagged -
    not just present, but present *as belonging to that frame*, so a later
    unmount produces a real milestone instead of an empty one."""

    async def test_a_mounted_skill_is_remounted_and_its_history_is_correctly_tagged(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        frame = await tree.mount("cmake.diagnose_configure", owner="user")
        wb.add_tagged_turn([(None, user("configure is failing")), (frame.id, assistant("mounted, digging in"))])
        wb.pin("cache_var", "OPENSSL_ROOT_DIR is unset", frame_id=frame.id)
        store = SessionStore(tmp_path)

        store.save(wb, frame_tree=tree)

        new_wb, new_tree = make_tree(skills_dir)
        result = await store.load(new_wb, new_tree)

        assert result.mounted_skills == ["cmake.diagnose_configure"]
        assert result.failed_skills == []
        assert new_tree.focused_skill_name == "cmake.diagnose_configure"

        new_frame = new_tree._frames[new_tree.focused_frame_id]  # whitebox, mirrors test_frames.py's own style
        assert new_frame.owner == "user"
        assert new_wb.tagged_turns == [[(None, user("configure is failing")), (new_frame.id, assistant("mounted, digging in"))]]
        assert new_wb.tagged_scratchpad == {"cache_var": (new_frame.id, "OPENSSL_ROOT_DIR is unset")}

        # the actual acceptance test: unmounting the restored frame distills
        # its REAL restored history, not an empty "no activity" milestone
        milestone = await new_tree.unmount("cmake.diagnose_configure", by="user")
        assert "no activity" not in milestone.lower()

    async def test_nested_frames_round_trip_with_correct_parent_child_structure(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        parent = await tree.mount("cmake.diagnose_configure")
        child = await tree.mount("cmake.inspect_cache")
        wb.add_tagged_turn([(parent.id, user("configure failed")), (child.id, assistant("checking the cache"))])
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        new_wb, new_tree = make_tree(skills_dir)
        result = await store.load(new_wb, new_tree)

        assert result.mounted_skills == ["cmake.diagnose_configure", "cmake.inspect_cache"]
        new_frames = {f.skill_name: f for f in new_tree.mounted_frames()}
        assert new_frames["cmake.inspect_cache"].parent_id == new_frames["cmake.diagnose_configure"].id
        assert new_tree.focused_skill_name == "cmake.inspect_cache"  # most-recently-mounted takes focus

    async def test_focus_is_restored_to_a_specific_mounted_skill(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        await tree.mount("cmake.diagnose_configure")
        await tree.mount("cmake.inspect_cache")
        tree.focus("cmake.diagnose_configure")  # step back to the parent before saving
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        new_wb, new_tree = make_tree(skills_dir)
        await store.load(new_wb, new_tree)

        assert new_tree.focused_skill_name == "cmake.diagnose_configure"

    async def test_focus_is_restored_to_root_even_when_skills_are_mounted(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        await tree.mount("cmake.diagnose_configure")
        tree.focus(None)  # explicitly stepped back to root before saving
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        new_wb, new_tree = make_tree(skills_dir)
        await store.load(new_wb, new_tree)

        assert new_tree.focused_skill_name is None

    async def test_pinned_flag_round_trips(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        await tree.mount("cmake.diagnose_configure", pinned=True)
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        new_wb, new_tree = make_tree(skills_dir)
        await store.load(new_wb, new_tree)

        restored_frame = new_tree.mounted_frames()[0]
        assert restored_frame.pinned is True

    async def test_a_removed_skill_is_reported_failed_and_its_history_falls_back_to_root(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        frame = await tree.mount("cmake.diagnose_configure")
        wb.add_tagged_turn([(frame.id, user("configure failed")), (frame.id, assistant("investigating"))])
        wb.pin("note", "was investigating configure", frame_id=frame.id)
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        # simulate the skill having been removed/renamed on disk since the save
        empty_skills_dir = tmp_path / "empty_skills"
        empty_skills_dir.mkdir()
        new_wb = ContextWorkbench()
        new_tree = FrameTree(new_wb, SkillStore(empty_skills_dir))

        result = await store.load(new_wb, new_tree)

        assert result.failed_skills == ["cmake.diagnose_configure"]
        assert result.mounted_skills == []
        # content is preserved, just no longer tagged to any frame
        assert new_wb.tagged_turns == [[(None, user("configure failed")), (None, assistant("investigating"))]]
        assert new_wb.tagged_scratchpad == {"note": (None, "was investigating configure")}

    async def test_a_missing_parent_lands_its_child_at_root_instead_of_failing_the_child_too(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        await tree.mount("cmake.diagnose_configure")
        await tree.mount("cmake.inspect_cache")  # nests under diagnose_configure
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        # only the parent skill goes missing before load
        partial_skills_dir = tmp_path / "partial_skills"
        write_skill(partial_skills_dir, "cmake/inspect_cache", "cmake.inspect_cache", "Inspect the cache.")
        new_wb = ContextWorkbench()
        new_tree = FrameTree(new_wb, SkillStore(partial_skills_dir))

        result = await store.load(new_wb, new_tree)

        assert result.failed_skills == ["cmake.diagnose_configure"]
        assert result.mounted_skills == ["cmake.inspect_cache"]
        restored_child = new_tree.mounted_frames()[0]
        assert restored_child.skill_name == "cmake.inspect_cache"
        assert restored_child.parent_id is None  # mounted at root, not silently dropped

    async def test_over_budget_at_load_time_is_reported_failed_not_raised(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        await tree.mount("cmake.diagnose_configure")
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        tiny_wb = ContextWorkbench(mounted_skill_budget=5)
        tiny_tree = FrameTree(tiny_wb, SkillStore(skills_dir))

        result = await store.load(tiny_wb, tiny_tree)

        assert result.failed_skills == ["cmake.diagnose_configure"]
        assert result.mounted_skills == []

    async def test_loading_with_no_frame_tree_falls_back_to_untagged_history_for_every_saved_frame(self, tmp_path, skills_dir):
        wb, tree = make_tree(skills_dir)
        frame = await tree.mount("cmake.diagnose_configure")
        wb.add_tagged_turn([(frame.id, user("hi"))])
        store = SessionStore(tmp_path)
        store.save(wb, frame_tree=tree)

        plain_wb = ContextWorkbench()
        result = await store.load(plain_wb)  # no frame_tree passed

        assert result.failed_skills == ["cmake.diagnose_configure"]
        assert plain_wb.turns == [[user("hi")]]

    async def test_load_clears_any_skills_already_mounted_in_the_target_tree(self, tmp_path, skills_dir):
        store = SessionStore(tmp_path)
        empty_wb, empty_tree = make_tree(skills_dir)
        store.save(empty_wb, frame_tree=empty_tree)  # a session with nothing mounted

        target_wb, target_tree = make_tree(skills_dir)
        await target_tree.mount("git.bisect")  # pre-existing mount, not part of the saved session

        await store.load(target_wb, target_tree)

        assert target_tree.mounted_frames() == []


class TestLoadReplacesExistingContent:
    async def test_load_clears_whatever_was_already_in_the_workbench(self, tmp_path):
        store = SessionStore(tmp_path)
        saved = ContextWorkbench()
        saved.add_turn([user("saved turn")])
        store.save(saved)

        target = ContextWorkbench()
        target.add_turn([user("unsaved, should be gone after load")])
        target.pin("stale", "pin")

        await store.load(target)

        assert target.turns == [[user("saved turn")]]
        assert target.scratchpad == {}


class TestErrorCases:
    async def test_loading_an_unknown_session_raises_session_not_found(self, tmp_path):
        store = SessionStore(tmp_path)
        with pytest.raises(SessionNotFound):
            await store.load(ContextWorkbench(), name="nope")

    @pytest.mark.parametrize("bad_name", ["../escape", "/etc/passwd", "..", ".", "a/b"])
    def test_unsafe_names_are_rejected_on_save(self, tmp_path, bad_name):
        store = SessionStore(tmp_path)
        with pytest.raises(UnsafeSessionName):
            store.save(ContextWorkbench(), name=bad_name)

    @pytest.mark.parametrize("bad_name", ["../escape", "/etc/passwd", "..", ".", "a/b"])
    async def test_unsafe_names_are_rejected_on_load(self, tmp_path, bad_name):
        store = SessionStore(tmp_path)
        with pytest.raises(UnsafeSessionName):
            await store.load(ContextWorkbench(), name=bad_name)


class TestListNames:
    def test_empty_root_lists_nothing(self, tmp_path):
        store = SessionStore(tmp_path / "does_not_exist_yet")
        assert store.list_names() == []

    def test_lists_every_saved_session_sorted(self, tmp_path):
        store = SessionStore(tmp_path)
        store.save(ContextWorkbench(), name="zeta")
        store.save(ContextWorkbench(), name="alpha")

        assert store.list_names() == ["alpha", "zeta"]


class TestAtomicWrite:
    def test_save_leaves_no_stray_temp_files_behind(self, tmp_path):
        store = SessionStore(tmp_path)
        store.save(ContextWorkbench(), name="s")

        assert [p.name for p in tmp_path.iterdir()] == ["s.json"]
