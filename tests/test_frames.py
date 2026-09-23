"""FrameTree: multiple and nested mounts, focus, subtree eviction, pinned
mounts, and the Mounted Skill Zone budget (ftw_plan.md §3.2 "Frames").

The central, explicitly-required invariant: unmounting a frame must return
the workbench to exactly its pre-mount state plus one milestone — nothing
belonging to that frame (or its descendants) leaks past the unmount.
"""

import pytest

from ftw.frames import (
    Frame,
    FrameBudgetExceeded,
    FrameNotFound,
    FramePinned,
    FrameTree,
    SkillAlreadyMounted,
    make_llm_summarizer,
)
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.skills.registry import SkillStore
from ftw.workbench import ContextWorkbench


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.USER, content=text)


def assistant_with_tool(name: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id="c1", name=name, arguments={})])


@pytest.fixture
def store(tmp_path):
    write_skill(tmp_path, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
    write_skill(tmp_path, "toolchain/verify_installed", "toolchain.verify_installed", "Verify a compiler toolchain.")
    write_skill(tmp_path, "git/bisect", "git.bisect", "Binary-search history for a regression.")
    return SkillStore(tmp_path)


@pytest.fixture
def workbench():
    return ContextWorkbench(system_anchor="anchor")


@pytest.fixture
def tree(workbench, store):
    return FrameTree(workbench, store)


class TestMount:
    def test_mounting_injects_skill_text_and_takes_focus(self, tree, workbench):
        frame = tree.mount("cmake.diagnose_configure", owner="user")

        assert isinstance(frame, Frame)
        assert frame.owner == "user"
        assert frame.parent_id is None
        assert workbench.mounted_skill_tokens > 0
        assert tree.focused_skill_name == "cmake.diagnose_configure"

    def test_mounting_unknown_skill_propagates_not_found(self, tree):
        from ftw.skills.registry import SkillNotFound

        with pytest.raises(SkillNotFound):
            tree.mount("nope.nothing")

    def test_mounting_same_skill_twice_raises(self, tree):
        tree.mount("cmake.diagnose_configure")
        with pytest.raises(SkillAlreadyMounted):
            tree.mount("cmake.diagnose_configure")

    def test_mounting_over_budget_is_refused_with_candidates_listed(self, store):
        wb = ContextWorkbench(mounted_skill_budget=5)  # tiny — any real skill body exceeds it
        tree = FrameTree(wb, store)
        with pytest.raises(FrameBudgetExceeded, match="cmake.diagnose_configure"):
            tree.mount("cmake.diagnose_configure")
        assert wb.mounted_skill_tokens == 0  # refused, not partially applied

    def test_model_mount_nests_under_current_focus(self, tree):
        tree.mount("cmake.diagnose_configure", owner="user")
        child = tree.mount("toolchain.verify_installed", owner="model")

        assert child.parent_id is not None
        parent = tree._frames[child.parent_id]  # noqa: SLF001 - whitebox check of the tree shape
        assert parent.skill_name == "cmake.diagnose_configure"
        assert tree.focused_skill_name == "toolchain.verify_installed"

    def test_focus_none_before_mount_gives_a_root_sibling(self, tree):
        tree.mount("cmake.diagnose_configure", owner="user")
        tree.focus(None)
        sibling = tree.mount("git.bisect", owner="user")

        assert sibling.parent_id is None


class TestFocusedFrameId:
    def test_none_when_nothing_mounted(self, tree):
        assert tree.focused_frame_id is None

    def test_matches_the_mounted_frames_id(self, tree):
        frame = tree.mount("cmake.diagnose_configure")
        assert tree.focused_frame_id == frame.id

    def test_updates_after_unmount(self, tree):
        parent = tree.mount("cmake.diagnose_configure")
        tree.mount("toolchain.verify_installed")
        tree.unmount("toolchain.verify_installed")
        assert tree.focused_frame_id == parent.id


class TestFocus:
    def test_focus_by_name_switches_focused_frame(self, tree):
        tree.mount("cmake.diagnose_configure")
        tree.focus(None)
        tree.mount("git.bisect")

        tree.focus("cmake.diagnose_configure")

        assert tree.focused_skill_name == "cmake.diagnose_configure"

    def test_focus_unknown_skill_raises(self, tree):
        with pytest.raises(FrameNotFound):
            tree.focus("nope.nothing")


class TestSingleFrameInvariant:
    """workbench_after_unmount == workbench_before_mount + milestone."""

    def test_unmount_restores_baseline_plus_one_milestone(self, tree, workbench):
        before_turns, before_scratchpad, before_milestones = workbench.turns, workbench.scratchpad, workbench.milestones

        frame = tree.mount("cmake.diagnose_configure", owner="user")
        workbench.add_turn([user("diagnose it"), assistant_with_tool("run_command")], frame_id=frame.id)
        workbench.pin("finding", "missing openssl", frame_id=frame.id)

        assert workbench.mounted_skill_tokens > 0  # sanity: mount actually changed the workbench

        milestone = tree.unmount("cmake.diagnose_configure", by="user")

        assert workbench.mounted_skill_tokens == 0
        assert workbench.turns == before_turns
        assert workbench.scratchpad == before_scratchpad
        assert workbench.milestones == before_milestones + [milestone]
        assert tree.focused_skill_name is None

    def test_unmount_unknown_skill_raises(self, tree):
        with pytest.raises(FrameNotFound):
            tree.unmount("nope.nothing")

    def test_deterministic_fallback_names_the_skill_and_tools_used(self, tree, workbench):
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go"), assistant_with_tool("run_command")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")

        assert "cmake.diagnose_configure" in milestone
        assert "run_command" in milestone

    def test_invariant_holds_with_a_non_empty_baseline_before_the_mount(self, tree, workbench):
        """The earlier version of this class only ever mounted into a
        workbench that started completely empty — "baseline" was
        vacuously []. Real sessions already have turns, pins, and
        milestones from before a skill is ever mounted; the invariant has
        to hold relative to whatever that baseline actually was, not just
        for the empty case."""
        workbench.add_turn([user("earlier, unrelated question")])
        workbench.pin("repo", "/home/user/src/core")
        workbench.add_milestone("an earlier, unrelated milestone")
        before_turns = list(workbench.turns)
        before_scratchpad = dict(workbench.scratchpad)
        before_milestones = list(workbench.milestones)

        frame = tree.mount("cmake.diagnose_configure", owner="user")
        workbench.add_turn([user("diagnose it"), assistant_with_tool("run_command")], frame_id=frame.id)
        workbench.pin("finding", "missing openssl", frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure", by="user")

        assert workbench.mounted_skill_tokens == 0
        assert workbench.turns == before_turns  # the baseline turn is untouched, in its original position
        assert workbench.scratchpad == before_scratchpad  # the baseline pin survives; only the frame's own pin is gone
        assert workbench.milestones == before_milestones + [milestone]  # baseline milestone kept, exactly one new one
        assert tree.focused_skill_name is None

    def test_invariant_holds_when_baseline_turns_were_already_evicted_by_ordinary_fifo_pressure(self, store):
        """A tight turn_horizon_budget can evict a baseline (unrelated)
        turn via ordinary FIFO pressure while a frame is mounted and
        working — nothing to do with the frame itself. unmount()'s own
        eviction must not get confused by that: it should remove exactly
        the frame's own surviving tagged content and nothing else, leaving
        whatever ordinary FIFO eviction already did alone."""
        workbench = ContextWorkbench(system_anchor="anchor", turn_horizon_budget=12)
        tree = FrameTree(workbench, store)

        workbench.add_turn([user("one two three four five")])  # baseline turn 1: 5 tokens
        workbench.add_turn([user("six seven eight nine ten")])  # baseline turn 2: 5 tokens

        frame = tree.mount("cmake.diagnose_configure", owner="user")
        workbench.add_turn([user("eleven twelve thirteen")], frame_id=frame.id)  # 3 tokens; 5+5+3=13 > 12

        # sanity: ordinary FIFO pressure, unrelated to the frame, already
        # evicted the oldest (baseline) turn before unmount ever runs
        assert workbench.turns == [[user("six seven eight nine ten")], [user("eleven twelve thirteen")]]

        milestone = tree.unmount("cmake.diagnose_configure", by="user")

        assert workbench.mounted_skill_tokens == 0
        assert workbench.turns == [[user("six seven eight nine ten")]]  # only the FIFO-surviving baseline turn remains
        assert workbench.milestones == [milestone]
        assert tree.focused_skill_name is None


class TestSiblingFrames:
    def test_unmounting_one_sibling_leaves_the_other_intact(self, tree, workbench):
        tree.focus(None)
        a = tree.mount("cmake.diagnose_configure")
        tree.focus(None)
        b = tree.mount("git.bisect")
        workbench.add_turn([user("a turn")], frame_id=a.id)
        workbench.add_turn([user("b turn")], frame_id=b.id)

        tree.unmount("cmake.diagnose_configure", by="user")

        assert workbench.turns == [[user("b turn")]]
        assert workbench.mounted_skill_tokens > 0  # git.bisect's body is still in the zone
        assert tree.focused_skill_name == "git.bisect" or tree.focused_skill_name is None


class TestNestedFrames:
    def test_unmounting_parent_evicts_the_whole_subtree_in_one_milestone(self, tree, workbench):
        before_milestones = workbench.milestones

        parent = tree.mount("cmake.diagnose_configure", owner="user")
        workbench.add_turn([user("diagnosing")], frame_id=parent.id)
        child = tree.mount("toolchain.verify_installed", owner="model")
        workbench.add_turn([user("checking toolchain"), assistant_with_tool("run_command")], frame_id=child.id)

        milestone = tree.unmount("cmake.diagnose_configure", by="user")

        assert workbench.mounted_skill_tokens == 0
        assert workbench.turns == []
        assert workbench.milestones == before_milestones + [milestone]  # exactly one new entry
        assert "toolchain.verify_installed" in milestone  # child's summary folded into the parent's
        assert tree.focused_skill_name is None

    def test_unmounting_a_child_directly_leaves_parent_mounted(self, tree, workbench):
        parent = tree.mount("cmake.diagnose_configure", owner="user")
        child = tree.mount("toolchain.verify_installed", owner="model")
        workbench.add_turn([user("child turn")], frame_id=child.id)
        workbench.add_turn([user("parent turn")], frame_id=parent.id)

        tree.unmount("toolchain.verify_installed", by="model")

        assert workbench.turns == [[user("parent turn")]]
        assert tree.focused_skill_name == "cmake.diagnose_configure"  # focus returns to the parent


class TestPinnedMounts:
    def test_model_cannot_unmount_a_pinned_frame(self, tree):
        tree.mount("cmake.diagnose_configure", owner="user", pinned=True)
        with pytest.raises(FramePinned):
            tree.unmount("cmake.diagnose_configure", by="model")

    def test_user_can_unmount_a_pinned_frame(self, tree):
        tree.mount("cmake.diagnose_configure", owner="user", pinned=True)
        tree.unmount("cmake.diagnose_configure", by="user")  # must not raise

    def test_model_cannot_unmount_an_ancestor_of_a_pinned_frame(self, tree):
        tree.mount("cmake.diagnose_configure", owner="user")
        tree.mount("toolchain.verify_installed", owner="model", pinned=True)
        with pytest.raises(FramePinned):
            tree.unmount("cmake.diagnose_configure", by="model")


class TestUnmountFocused:
    def test_unmount_focused_targets_the_focused_frame(self, tree, workbench):
        tree.mount("cmake.diagnose_configure")
        milestone = tree.unmount_focused(by="user")
        assert "cmake.diagnose_configure" in milestone
        assert workbench.mounted_skill_tokens == 0

    def test_unmount_focused_with_nothing_mounted_raises(self, tree):
        with pytest.raises(FrameNotFound):
            tree.unmount_focused()


class TestSummarizer:
    def test_llm_summarizer_result_is_used_when_it_succeeds(self, store, workbench):
        tree = FrameTree(workbench, store, summarizer=lambda manifest, transcript: "LLM: found the bug")
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")

        assert milestone == "LLM: found the bug"

    def test_falls_back_to_deterministic_when_summarizer_returns_none(self, store, workbench):
        tree = FrameTree(workbench, store, summarizer=lambda manifest, transcript: None)
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go"), assistant_with_tool("run_command")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")

        assert "cmake.diagnose_configure" in milestone
        assert "run_command" in milestone

    def test_falls_back_to_deterministic_when_summarizer_raises(self, store, workbench):
        def boom(manifest, transcript):
            raise RuntimeError("provider unreachable")

        tree = FrameTree(workbench, store, summarizer=boom)
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")  # must not raise

        assert "cmake.diagnose_configure" in milestone


class TestMakeLlmSummarizer:
    def test_uses_the_providers_response_text(self, store, workbench):
        provider = MockModelProvider(
            [ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content="Fixed the missing OpenSSL headers."))]
        )
        tree = FrameTree(workbench, store, summarizer=make_llm_summarizer(provider))
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")

        assert milestone == "Fixed the missing OpenSSL headers."

    def test_falls_back_when_the_provider_raises(self, store, workbench):
        class BoomProvider:
            def complete(self, messages, tools=None):
                raise RuntimeError("network down")

        tree = FrameTree(workbench, store, summarizer=make_llm_summarizer(BoomProvider()))
        frame = tree.mount("cmake.diagnose_configure")
        workbench.add_turn([user("go"), assistant_with_tool("run_command")], frame_id=frame.id)

        milestone = tree.unmount("cmake.diagnose_configure")

        assert "cmake.diagnose_configure" in milestone
        assert "run_command" in milestone


class TestFindAndRenderTree:
    def test_find_delegates_to_the_skill_store(self, tree):
        assert tree.find("cmake configuration")[0] == "cmake.diagnose_configure"

    def test_render_tree_shows_nesting_and_focus(self, tree):
        parent = tree.mount("cmake.diagnose_configure", owner="user")
        tree.mount("toolchain.verify_installed", owner="model")

        text = tree.render_tree()

        assert "cmake.diagnose_configure" in text
        assert "toolchain.verify_installed" in text
        assert text.index("cmake.diagnose_configure") < text.index("toolchain.verify_installed")

    def test_render_tree_on_empty_tree(self, tree):
        assert "root" in tree.render_tree().lower()
