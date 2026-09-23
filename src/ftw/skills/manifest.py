"""SKILL.md manifest schema, parsing, and the body size lint
(ftw_plan.md §3.1).

A SKILL.md file is YAML frontmatter (metadata: name, version, description,
inputs/outputs, calls allowlist, budget, verify) followed by a Markdown
body — the actual procedure text. Only the body counts against the
1,500-token cap and only the body (plus the header render_frame_text adds)
is what gets injected into the Mounted Skill zone; the frontmatter is
read by the framework, never sent to the model verbatim.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ftw.tokens import count_tokens

MAX_SKILL_BODY_TOKENS = 1500

_FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?\n)---\s*\n?(.*)\Z", re.DOTALL)


class SkillParseError(Exception):
    pass


class SkillTooLarge(SkillParseError):
    pass


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = 15
    max_tokens: int = 25000
    deadline_ms: int = 120_000


class NeedsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: list[str] = []


class VerifyCheck(BaseModel):
    """Deterministic checks and critic prompts are different shapes; both
    are optional fields here since Phase 2 only needs to parse and carry
    this forward — enforcing it is Phase 4's Verifier pipeline."""

    model_config = ConfigDict(extra="forbid")

    check: str | None = None
    cmd: str | None = None
    expect_exit: int | None = None
    after_fix: bool | None = None
    critic: str | None = None


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    version: int = 1
    kind: Literal["prompt", "tool", "composite"] = "prompt"
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    calls: list[str] = []
    needs: NeedsSpec = NeedsSpec()
    model: str | None = None
    budget: BudgetSpec = BudgetSpec()
    verify: list[VerifyCheck] = []


def parse_skill_md(text: str) -> tuple[SkillManifest, str]:
    """Splits a SKILL.md file into its manifest and body."""
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        raise SkillParseError("SKILL.md must open with '---' YAML frontmatter closed by a matching '---'")

    frontmatter_text, body = match.group(1), match.group(2)

    try:
        data: Any = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(f"invalid YAML frontmatter: {exc}") from exc

    try:
        manifest = SkillManifest.model_validate(data)
    except ValidationError as exc:
        raise SkillParseError(f"invalid SKILL.md manifest: {exc}") from exc

    return manifest, body.strip()


def lint_skill_body(body: str, *, max_tokens: int = MAX_SKILL_BODY_TOKENS) -> None:
    """Raises SkillTooLarge if ``body`` exceeds the token cap. A skill that
    nears this threshold must be split or refactored, not silently truncated."""
    n = count_tokens(body)
    if n > max_tokens:
        raise SkillTooLarge(f"skill body is {n} tokens, exceeds the {max_tokens}-token cap")


def render_frame_text(manifest: SkillManifest, body: str) -> str:
    """The text a mounted skill contributes to the Mounted Skill zone."""
    return f"## Skill: {manifest.name} (v{manifest.version})\n{manifest.description}\n\n{body}"
