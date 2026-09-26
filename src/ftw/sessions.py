"""SessionStore: save/load a ContextWorkbench — and, if given, a FrameTree
— to/from a JSON file (ftw_plan.md §3.6 ``/save``, ``/load``). "Files are
the source of truth" (CLAUDE.md): a saved session is a plain, inspectable
JSON file under ``$FTW_HOME/sessions``, not an opaque blob.

**Frame fidelity.** A mounted frame's id (``Frame.id``, ``frames.py``) is a
fresh UUID minted every time it's mounted — never stable across a process
restart. What *is* stable is its ``skill_name``: ``FrameTree.mount()``
already refuses to mount the same skill twice at once
(``SkillAlreadyMounted``), so at any moment a skill_name uniquely
identifies a mounted frame. Saving serializes each currently-mounted frame
by that name (plus its parent's name, for nesting) instead of its id, and
translates every tagged turn/pin's frame_id through the same name. Loading
remounts each saved frame — in saved order, so a parent is always remounted
before the child that nests under it — through the *real*
``FrameTree.mount()``, which is what makes this correct rather than a
best-effort approximation: the resulting tree, focus, and Mounted Skill
zone are built by the exact same code a live mount/mount would build them
with, not reconstructed by hand. The new skill_name -> new-id mapping that
falls out of that is then used to re-tag every restored turn/pin, so
``ContextWorkbench.evict_frame`` (what an ``unmount`` after loading relies
on) has exactly the same information it would have had if the process had
never restarted at all.

**Partial failure never drops content.** A saved frame can fail to remount
— the skill was renamed/removed on disk, or today's Mounted Skill budget
doesn't fit it. That frame's name is reported in ``LoadResult.failed_skills``
rather than raising, and anything tagged to it (turns, pins) falls back to
the root session (frame_id ``None``) instead of being discarded — the
message content always survives, even when its frame doesn't. A child
whose parent failed to remount is mounted at the root instead of also
failing, so one missing skill doesn't cascade into losing its siblings'
skills too.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ftw.frames import FrameError, FrameTree
from ftw.providers import ChatMessage
from ftw.skills.registry import SkillNotFound
from ftw.workbench import ContextWorkbench

DEFAULT_SESSION_NAME = "latest"

# A session name reaches _path() straight from a slash-command argument,
# so it's checked the same way outputs.py's _SAFE_ID checks an
# output_id — no path separators, no "..", not absolute.
_SAFE_NAME = re.compile(r"\A[A-Za-z0-9._-]+\Z")


class SessionNotFound(Exception):
    pass


class UnsafeSessionName(ValueError):
    pass


def _check_safe(name: str) -> None:
    if not _SAFE_NAME.match(name) or name in (".", ".."):
        raise UnsafeSessionName(f"unsafe session name: {name!r}")


@dataclass
class LoadResult:
    turn_count: int
    mounted_skills: list[str]
    failed_skills: list[str]


class SessionStore:
    def __init__(self, root: str | Path):
        self._root = Path(root)

    def save(self, workbench: ContextWorkbench, frame_tree: FrameTree | None = None, name: str = DEFAULT_SESSION_NAME) -> Path:
        path = self._path(name)

        frame_id_to_skill: dict[str, str] = {}
        frames_data: list[dict[str, Any]] = []
        focused_skill: str | None = None
        if frame_tree is not None:
            mounted = frame_tree.mounted_frames()
            frame_id_to_skill = {f.id: f.skill_name for f in mounted}
            for f in mounted:
                parent_skill_name = frame_id_to_skill.get(f.parent_id) if f.parent_id is not None else None
                frames_data.append(
                    {
                        "skill_name": f.skill_name,
                        "owner": f.owner,
                        "pinned": f.pinned,
                        "parent_skill_name": parent_skill_name,
                    }
                )
            focused_skill = frame_tree.focused_skill_name

        def label(frame_id: str | None) -> str | None:
            return frame_id_to_skill.get(frame_id) if frame_id is not None else None

        data = {
            "version": 2,
            "saved_at": datetime.now(UTC).isoformat(),
            "focused_skill": focused_skill,
            "frames": frames_data,
            "milestones": workbench.milestones,
            "scratchpad": [
                {"key": key, "value": value, "skill": label(frame_id)}
                for key, (frame_id, value) in workbench.tagged_scratchpad.items()
            ],
            "turns": [
                [{"skill": label(frame_id), "message": message.model_dump(mode="json")} for frame_id, message in turn]
                for turn in workbench.tagged_turns
            ],
        }
        self._root.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(path)  # atomic on POSIX — a crash mid-write never leaves a half-written session
        return path

    async def load(
        self, workbench: ContextWorkbench, frame_tree: FrameTree | None = None, name: str = DEFAULT_SESSION_NAME
    ) -> LoadResult:
        """Replaces whatever's currently in ``workbench`` (and, if given,
        currently mounted in ``frame_tree``) with the saved session's
        state. See the module docstring for what "correctly" means here."""
        path = self._path(name)
        if not path.exists():
            raise SessionNotFound(f"no saved session named {name!r}")
        data = json.loads(path.read_text())

        workbench.reset()
        if frame_tree is not None:
            frame_tree.clear()

        for milestone in data.get("milestones", []):
            workbench.add_milestone(milestone)

        skill_to_new_id: dict[str, str] = {}
        mounted_skills: list[str] = []
        failed_skills: list[str] = []

        frames = data.get("frames", [])
        if frame_tree is not None:
            for spec in frames:
                skill_name = spec["skill_name"]
                parent_skill_name = spec.get("parent_skill_name")
                if parent_skill_name is not None and parent_skill_name not in skill_to_new_id:
                    parent_skill_name = None  # parent didn't remount (or wasn't recorded) - land at root instead
                frame_tree.focus(parent_skill_name)
                try:
                    frame = await frame_tree.mount(skill_name, owner=spec.get("owner", "model"), pinned=spec.get("pinned", False))
                except (FrameError, SkillNotFound):
                    failed_skills.append(skill_name)
                    continue
                skill_to_new_id[skill_name] = frame.id
                mounted_skills.append(skill_name)

            saved_focus = data.get("focused_skill")
            frame_tree.focus(saved_focus if saved_focus in skill_to_new_id else None)
        else:
            failed_skills = [spec["skill_name"] for spec in frames]

        def resolve(skill_name: str | None) -> str | None:
            return skill_to_new_id.get(skill_name) if skill_name is not None else None

        for item in data.get("scratchpad", []):
            workbench.pin(item["key"], item["value"], resolve(item.get("skill")))

        turns = data.get("turns", [])
        for turn in turns:
            tagged = [(resolve(entry.get("skill")), ChatMessage.model_validate(entry["message"])) for entry in turn]
            workbench.add_tagged_turn(tagged)

        return LoadResult(turn_count=len(turns), mounted_skills=mounted_skills, failed_skills=failed_skills)

    def list_names(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json"))

    def _path(self, name: str) -> Path:
        _check_safe(name)
        return self._root / f"{name}.json"
