"""ContextWorkbench: zoned token budgets, ordering, and prompt rendering
(ftw_plan.md §3.3).

Zone order in the rendered prompt is deliberately least-volatile-first,
most-volatile-last (System -> User -> MountedSkill -> Milestones ->
TurnHorizon -> Scratchpad) so KV/prefix caching on local runtimes and
providers isn't invalidated by the parts of the prompt that change every
turn.
"""

import pytest

from ftw.providers import ChatMessage, ChatRole
from ftw.workbench import ContextWorkbench, WorkbenchBudgetExceeded


def user(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.USER, content=text)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role=ChatRole.ASSISTANT, content=text)


class TestStaticZones:
    def test_system_and_user_memory_tokens_counted_at_construction(self):
        wb = ContextWorkbench(system_anchor="be terse and safe", user_memory="prefers uv, no PRs")
        assert wb.system_anchor_tokens > 0
        assert wb.user_memory_tokens > 0

    def test_defaults_to_empty_zones(self):
        wb = ContextWorkbench()
        assert wb.system_anchor_tokens == 0
        assert wb.user_memory_tokens == 0
        assert wb.mounted_skill_tokens == 0
        assert wb.milestones_tokens == 0
        assert wb.turn_horizon_tokens == 0
        assert wb.scratchpad_tokens == 0


class TestMountedSkillHook:
    def test_set_mounted_skill_text_updates_token_count(self):
        wb = ContextWorkbench()
        wb.set_mounted_skill_text("skill: cmake.diagnose_configure\n...")
        assert wb.mounted_skill_tokens > 0

    def test_clearing_mounted_skill_text_returns_to_zero(self):
        wb = ContextWorkbench()
        wb.set_mounted_skill_text("some skill body")
        wb.set_mounted_skill_text("")
        assert wb.mounted_skill_tokens == 0


class TestMilestones:
    def test_add_milestone_increases_tokens(self):
        wb = ContextWorkbench()
        wb.add_milestone("Diagnosed missing OpenSSL headers; patched CMakeLists.txt.")
        assert wb.milestones_tokens > 0
        assert wb.milestones == ["Diagnosed missing OpenSSL headers; patched CMakeLists.txt."]

    def test_oldest_milestone_evicted_when_budget_exceeded(self):
        wb = ContextWorkbench(milestones_budget=10)
        wb.add_milestone("one two three four five six seven eight nine ten")
        wb.add_milestone("eleven twelve thirteen fourteen fifteen sixteen")
        assert wb.milestones_tokens <= 20
        assert "one two three four five six seven eight nine ten" not in wb.milestones


class TestScratchpadPins:
    def test_pin_and_unpin(self):
        wb = ContextWorkbench()
        wb.pin("repo", "/home/user/src/core")
        assert wb.scratchpad_tokens > 0
        assert wb.scratchpad["repo"] == "/home/user/src/core"

        wb.unpin("repo")
        assert wb.scratchpad_tokens == 0
        assert "repo" not in wb.scratchpad

    def test_pin_over_budget_raises_and_does_not_partially_apply(self):
        wb = ContextWorkbench(scratchpad_budget=5)
        with pytest.raises(WorkbenchBudgetExceeded):
            wb.pin("error", "a very long error message that blows past five tokens easily")
        assert wb.scratchpad_tokens == 0

    def test_unpin_missing_key_raises_key_error(self):
        wb = ContextWorkbench()
        with pytest.raises(KeyError):
            wb.unpin("nope")

    def test_repinning_same_key_replaces_value(self):
        wb = ContextWorkbench()
        wb.pin("target", "first")
        wb.pin("target", "second")
        assert wb.scratchpad["target"] == "second"
        assert len(wb.scratchpad) == 1


class TestTurnHorizon:
    def test_add_turn_appends_messages(self):
        wb = ContextWorkbench()
        wb.add_turn([user("hi"), assistant("hello")])
        assert wb.turn_horizon_tokens > 0
        assert len(wb.turns) == 1

    def test_oldest_turn_evicted_when_budget_exceeded(self):
        wb = ContextWorkbench(turn_horizon_budget=8)
        wb.add_turn([user("one two three four five")])
        wb.add_turn([user("six seven eight nine ten")])
        assert wb.turn_horizon_tokens <= 15
        assert len(wb.turns) == 1

    def test_eviction_invokes_callback_with_evicted_turn(self):
        evicted = []
        wb = ContextWorkbench(turn_horizon_budget=8, on_turn_evicted=evicted.append)
        first_turn = [user("one two three four five")]
        wb.add_turn(first_turn)
        wb.add_turn([user("six seven eight nine ten")])

        assert len(evicted) == 1
        assert evicted[0] == first_turn

    def test_clear_turns_empties_horizon_and_fires_callback_for_each(self):
        evicted = []
        wb = ContextWorkbench(on_turn_evicted=evicted.append)
        wb.add_turn([user("a")])
        wb.add_turn([user("b")])

        wb.clear_turns()

        assert wb.turns == []
        assert wb.turn_horizon_tokens == 0
        assert len(evicted) == 2


class TestRenderPrompt:
    def test_zone_order_in_system_message(self):
        wb = ContextWorkbench(system_anchor="ANCHOR", user_memory="MEMORY")
        wb.set_mounted_skill_text("SKILLTEXT")
        wb.add_milestone("MILESTONE-ONE")

        prompt = wb.render_prompt()
        system_msg = prompt[0]

        assert system_msg.role == ChatRole.SYSTEM
        content = system_msg.content
        assert content.index("ANCHOR") < content.index("MEMORY")
        assert content.index("MEMORY") < content.index("SKILLTEXT")
        assert content.index("SKILLTEXT") < content.index("MILESTONE-ONE")

    def test_turn_horizon_messages_follow_system_message(self):
        wb = ContextWorkbench(system_anchor="ANCHOR")
        wb.add_turn([user("first user turn"), assistant("first reply")])
        wb.add_turn([user("second user turn")])

        prompt = wb.render_prompt()

        assert prompt[0].role == ChatRole.SYSTEM
        assert prompt[1] == user("first user turn")
        assert prompt[2] == assistant("first reply")
        assert prompt[3] == user("second user turn")

    def test_scratchpad_is_last_message_when_present(self):
        wb = ContextWorkbench(system_anchor="ANCHOR")
        wb.add_turn([user("hi")])
        wb.pin("repo", "/home/user/src/core")

        prompt = wb.render_prompt()

        assert prompt[-1].role == ChatRole.SYSTEM
        assert "/home/user/src/core" in prompt[-1].content

    def test_scratchpad_message_omitted_when_empty(self):
        wb = ContextWorkbench(system_anchor="ANCHOR")
        wb.add_turn([user("hi")])

        prompt = wb.render_prompt()

        assert prompt[-1] == user("hi")

    def test_empty_workbench_renders_no_system_message(self):
        wb = ContextWorkbench()
        assert wb.render_prompt() == []

    def test_extra_messages_are_inserted_before_scratchpad(self):
        wb = ContextWorkbench(system_anchor="ANCHOR")
        wb.add_turn([user("committed turn")])
        wb.pin("repo", "/src")

        prompt = wb.render_prompt(extra_messages=[user("in-progress turn")])

        assert prompt[-1].role == ChatRole.SYSTEM  # scratchpad still last
        assert "/src" in prompt[-1].content
        assert prompt[-2] == user("in-progress turn")

    def test_extra_messages_appended_at_end_when_no_scratchpad(self):
        wb = ContextWorkbench(system_anchor="ANCHOR")
        wb.add_turn([user("committed turn")])

        prompt = wb.render_prompt(extra_messages=[user("in-progress turn")])

        assert prompt[-1] == user("in-progress turn")


class TestFrameTagging:
    """Every turn and pin is tagged with the frame it belongs to (ftw_plan.md
    §3.2 Frames), so unmounting a frame can evict exactly its own content —
    and nothing else — from the Turn Horizon and Scratchpad."""

    def test_add_turn_defaults_to_no_frame(self):
        wb = ContextWorkbench()
        wb.add_turn([user("hi")])
        assert wb.turns == [[user("hi")]]

    def test_evict_frame_removes_only_that_frames_turns_in_order(self):
        wb = ContextWorkbench()
        wb.add_turn([user("base turn")])
        wb.add_turn([user("frame turn 1")], frame_id="f1")
        wb.add_turn([user("other frame")], frame_id="f2")
        wb.add_turn([user("frame turn 2")], frame_id="f1")

        evicted = wb.evict_frame("f1")

        assert evicted == [[user("frame turn 1")], [user("frame turn 2")]]
        assert wb.turns == [[user("base turn")], [user("other frame")]]

    def test_evict_frame_with_nothing_tagged_returns_empty(self):
        wb = ContextWorkbench()
        wb.add_turn([user("hi")])
        assert wb.evict_frame("nope") == []
        assert wb.turns == [[user("hi")]]

    def test_evict_frame_fires_on_turn_evicted_callback(self):
        evicted = []
        wb = ContextWorkbench(on_turn_evicted=evicted.append)
        turn = [user("frame turn")]
        wb.add_turn(turn, frame_id="f1")

        wb.evict_frame("f1")

        assert evicted == [turn]

    def test_pin_defaults_to_no_frame(self):
        wb = ContextWorkbench()
        wb.pin("k", "v")
        assert wb.scratchpad == {"k": "v"}

    def test_evict_frame_pins_removes_only_that_frames_pins(self):
        wb = ContextWorkbench()
        wb.pin("base", "b", frame_id=None)
        wb.pin("mine", "m", frame_id="f1")
        wb.pin("other", "o", frame_id="f2")

        evicted = wb.evict_frame_pins("f1")

        assert evicted == {"mine": "m"}
        assert wb.scratchpad == {"base": "b", "other": "o"}

    def test_mounted_skill_budget_is_readable(self):
        wb = ContextWorkbench(mounted_skill_budget=1234)
        assert wb.mounted_skill_budget == 1234


class TestPerMessageTagging:
    """add_turn tags a WHOLE turn with one frame_id — fine when a mount
    and its unmount land in separate turns, but the model's natural
    self-mount pattern is mount -> work -> unmount all within one reply,
    committed as a single turn. Tagging the whole thing with whatever's
    focused at commit time (i.e. after the unmount) meant the frame's own
    content was never evicted at all. add_tagged_turn tags each message
    individually, so evict_frame can remove exactly the messages produced
    while that frame had focus and leave the rest of the turn alone."""

    def test_add_tagged_turn_basic_round_trip(self):
        wb = ContextWorkbench()
        wb.add_tagged_turn([(None, user("hi")), ("f1", assistant("mounted and worked"))])
        assert wb.turns == [[user("hi"), assistant("mounted and worked")]]

    def test_evict_frame_removes_only_the_tagged_messages_within_a_turn(self):
        wb = ContextWorkbench()
        wb.add_tagged_turn(
            [
                (None, user("diagnose it")),
                (None, assistant("mounting the skill")),  # decision, made before the mount takes effect
                ("f1", ChatMessage(role=ChatRole.TOOL, tool_call_id="c1", name="mount_skill", content="mounted")),
                ("f1", ChatMessage(role=ChatRole.TOOL, tool_call_id="c2", name="run_command", content="did the work")),
                (None, ChatMessage(role=ChatRole.TOOL, tool_call_id="c3", name="unmount_skill", content="unmounted")),
                (None, assistant("all done")),
            ]
        )

        evicted = wb.evict_frame("f1")

        assert len(evicted) == 1
        evicted_contents = [m.content for m in evicted[0]]
        assert evicted_contents == ["mounted", "did the work"]
        # everything NOT tagged to f1 survives, in its original order,
        # even though it was in the same turn as the evicted messages
        remaining_contents = [m.content for m in wb.turns[0]]
        assert remaining_contents == ["diagnose it", "mounting the skill", "unmounted", "all done"]

    def test_a_turn_fully_tagged_to_one_frame_is_removed_entirely(self):
        wb = ContextWorkbench()
        wb.add_tagged_turn([("f1", user("hi")), ("f1", assistant("bye"))])
        wb.evict_frame("f1")
        assert wb.turns == []

    def test_add_turn_is_add_tagged_turn_with_one_frame_id_for_every_message(self):
        """Backward-compatible convenience wrapper, still used wherever a
        whole turn genuinely belongs to one frame (or none)."""
        wb = ContextWorkbench()
        wb.add_turn([user("a"), assistant("b")], frame_id="f1")
        assert wb.evict_frame("f1") == [[user("a"), assistant("b")]]

    def test_partial_eviction_still_fires_on_turn_evicted_with_just_the_evicted_messages(self):
        evicted = []
        wb = ContextWorkbench(on_turn_evicted=evicted.append)
        wb.add_tagged_turn([(None, user("keep")), ("f1", assistant("evict me"))])

        wb.evict_frame("f1")

        assert len(evicted) == 1
        assert [m.content for m in evicted[0]] == ["evict me"]

    def test_turn_horizon_tokens_reflect_partial_eviction(self):
        wb = ContextWorkbench()
        wb.add_tagged_turn([(None, user("keep this")), ("f1", assistant("a rather long message to evict"))])
        before = wb.turn_horizon_tokens
        wb.evict_frame("f1")
        after = wb.turn_horizon_tokens
        assert after < before
        assert after == count_tokens_of_message(user("keep this"))


def count_tokens_of_message(message: ChatMessage) -> int:
    from ftw.tokens import count_tokens

    return count_tokens(message.content or "")


class TestSnapshot:
    def test_snapshot_reports_each_zone(self):
        wb = ContextWorkbench(system_anchor="anchor text", user_memory="memory text")
        wb.add_milestone("did a thing")
        wb.add_turn([user("hello")])
        wb.pin("k", "v")

        snap = wb.snapshot()
        names = {z.name for z in snap.zones}

        assert names == {"System Anchor", "User Memory", "Mounted Skill", "Milestones", "Turn Horizon", "Scratchpad"}
        by_name = {z.name: z for z in snap.zones}
        assert by_name["System Anchor"].tokens == wb.system_anchor_tokens
        assert by_name["Turn Horizon"].tokens == wb.turn_horizon_tokens
        assert snap.total_tokens == sum(z.tokens for z in snap.zones)

    def test_render_report_produces_readable_text(self):
        wb = ContextWorkbench(system_anchor="anchor", max_total_tokens=8000)
        report = wb.render_report()
        assert "System Anchor" in report
        assert "8000" in report or "8,000" in report
