"""Worker supervision and end-to-end wiring (ftw_plan.md §7 Phase 1
deliverable: "REPL mounts... runs confirmed shell commands over NNG").

This is the one place a real subprocess and a real ipc:// socket are
exercised — everything else in the suite stays over inproc://. Still zero
network calls and zero live LLM tokens: the model side is MockModelProvider.
"""

import io
import sys

from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.repl.cli import build_repl_session


class ScriptedInput:
    """Mimics input(): pops the next scripted line, raises EOFError once
    exhausted."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def __call__(self, prompt: str = "") -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def assistant_run_command() -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", name="run_command", arguments={"argv": [sys.executable, "-c", "print('from the real worker')"]})],
        )
    )


def assistant_text(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


class TestBuildReplSessionWithRealWorker:
    def test_shell_command_round_trips_through_a_real_subprocess_worker(self, tmp_path):
        provider = MockModelProvider([assistant_run_command(), assistant_text("it printed the line")])
        out = io.StringIO()

        handle = build_repl_session(
            ftw_home=tmp_path,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the one shell command
            output=out,
        )
        try:
            reply = handle.session.agent_loop.run_turn("run something")
        finally:
            handle.close()

        assert reply == "it printed the line"


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def tool_call_response(name: str, arguments: dict, call_id: str = "c") -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])
    )


def text_response(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


class TestPhase2Deliverable:
    """ftw_plan.md §8 Phase 2: "The model finds and mounts a skill (and a
    nested helper), works multi-turn with shell tools, and unmounts.
    /context shows the footprint back at baseline plus milestones." Proven
    against a real subprocess shell worker and real SKILL.md files on
    disk, across two genuinely separate turns — only the model side is
    mocked (and note that the *same* mocked provider also has to answer
    the milestone summarizer's calls, in the order they actually happen:
    two more completions, child-then-parent, fired from inside the
    unmount_skill tool call itself, before the turn's final answer)."""

    def test_find_mount_nested_mount_run_shell_then_unmount_restores_baseline(self, tmp_path):
        skills_dir = tmp_path / "skills"
        write_skill(skills_dir, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
        write_skill(skills_dir, "toolchain/verify_installed", "toolchain.verify_installed", "Verify a compiler toolchain.")

        provider = MockModelProvider(
            [
                # --- turn 1: find, mount (nested), run a confirmed shell command, answer ---
                tool_call_response("find_skill", {"query": "cmake configuration"}, "c1"),
                tool_call_response("mount_skill", {"name": "cmake.diagnose_configure"}, "c2"),
                tool_call_response("mount_skill", {"name": "toolchain.verify_installed"}, "c3"),
                tool_call_response(
                    "run_command", {"argv": [sys.executable, "-c", "print('gcc found')"]}, "c4"
                ),
                text_response("mounted cmake help and confirmed gcc is on PATH"),
                # --- turn 2: unmount (triggers two summarizer calls, child then parent), answer ---
                tool_call_response("unmount_skill", {"name": "cmake.diagnose_configure"}, "c5"),
                text_response("toolchain.verify_installed: gcc found on PATH."),  # summarizer call, child
                text_response("cmake.diagnose_configure: root cause identified."),  # summarizer call, parent
                text_response("all done"),
            ]
        )
        out = io.StringIO()
        handle = build_repl_session(
            ftw_home=tmp_path,
            skills_dir=skills_dir,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the one shell command
            output=out,
        )
        workbench = handle.session.agent_loop.workbench

        try:
            first_reply = handle.session.agent_loop.run_turn("diagnose my cmake failure")
            assert workbench.mounted_skill_tokens > 0  # still mounted between turns
            assert workbench.milestones == []

            second_reply = handle.session.agent_loop.run_turn("looks good, wrap it up")
        finally:
            handle.close()

        assert first_reply == "mounted cmake help and confirmed gcc is on PATH"
        assert second_reply == "all done"

        assert workbench.mounted_skill_tokens == 0  # the whole subtree was evicted
        assert len(workbench.milestones) == 1  # one combined milestone, child folded into parent
        assert workbench.milestones[0] == "cmake.diagnose_configure: root cause identified. [toolchain.verify_installed: gcc found on PATH.]"
        # turn 1 (tagged to the mounted frame) was evicted; turn 2 (committed after
        # the unmount, with nothing focused) remains, untagged, at the root
        assert len(workbench.turns) == 1
        assert workbench.turns[0][-1].content == "all done"

        snapshot = workbench.snapshot()
        by_name = {z.name: z for z in snapshot.zones}
        assert by_name["Mounted Skill"].tokens == 0
        assert by_name["Milestones"].detail == "1 milestone(s)"
