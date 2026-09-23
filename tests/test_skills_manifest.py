"""SKILL.md parsing, manifest validation, and the 1,500-token body cap
(ftw_plan.md §3.1).

The cap applies to the body — the procedure text that actually gets
injected into the Mounted Skill zone — not the YAML frontmatter, which is
metadata read by the framework, never sent to the model verbatim.
"""

import pytest

from ftw.skills.manifest import (
    MAX_SKILL_BODY_TOKENS,
    SkillManifest,
    SkillParseError,
    SkillTooLarge,
    lint_skill_body,
    parse_skill_md,
    render_frame_text,
)

VALID_SKILL_MD = """---
name: cmake.diagnose_configure
version: 1
description: Diagnose failing CMake configuration and isolate root causes with evidence.
kind: prompt
inputs:
  repo_dir: string
  error_text: string
outputs:
  cause: string
  fix: string
calls: [tool.shell, toolchain.verify_installed]
needs:
  env: [host.current, host.compilers]
model: fast
budget:
  max_steps: 15
  max_tokens: 25000
  deadline_ms: 120000
verify:
  - check: command_ok
    cmd: "cmake -S {repo_dir} -B /tmp/ftw-verify"
    expect_exit: 0
  - critic: "Does the cited build log confirm the missing dependency?"
---

# Diagnose Configure

1. Reproduce the failure by running `cmake -S {repo_dir} -B build`.
2. Grep the CMakeError.log for known dependency signatures.
3. Report the cause and a proposed fix with evidence.
"""


class TestParseSkillMd:
    def test_parses_frontmatter_into_a_manifest(self):
        manifest, body = parse_skill_md(VALID_SKILL_MD)

        assert isinstance(manifest, SkillManifest)
        assert manifest.name == "cmake.diagnose_configure"
        assert manifest.version == 1
        assert manifest.kind == "prompt"
        assert manifest.calls == ["tool.shell", "toolchain.verify_installed"]
        assert manifest.needs.env == ["host.current", "host.compilers"]
        assert manifest.budget.max_steps == 15
        assert len(manifest.verify) == 2

    def test_body_is_the_markdown_after_the_frontmatter(self):
        _manifest, body = parse_skill_md(VALID_SKILL_MD)
        assert body.startswith("# Diagnose Configure")
        assert "Report the cause" in body

    def test_minimal_manifest_uses_defaults(self):
        text = """---
name: demo.greet
description: Say hello.
---

Say hello to the user.
"""
        manifest, body = parse_skill_md(text)
        assert manifest.version == 1
        assert manifest.kind == "prompt"
        assert manifest.calls == []
        assert manifest.needs.env == []
        assert manifest.budget.max_steps == 15  # BudgetSpec default
        assert body.strip() == "Say hello to the user."

    def test_missing_frontmatter_delimiters_raises(self):
        with pytest.raises(SkillParseError):
            parse_skill_md("name: demo.greet\ndescription: hi\n\nbody text")

    def test_missing_required_field_raises(self):
        text = """---
version: 1
---

body
"""
        with pytest.raises(SkillParseError):
            parse_skill_md(text)  # no name, no description

    def test_invalid_yaml_raises_skill_parse_error(self):
        text = """---
name: [unterminated
---

body
"""
        with pytest.raises(SkillParseError):
            parse_skill_md(text)


class TestLintSkillBody:
    def test_short_body_passes(self):
        lint_skill_body("a short procedure body")  # must not raise

    def test_body_at_exactly_the_cap_passes(self):
        body = " ".join(["word"] * MAX_SKILL_BODY_TOKENS)
        lint_skill_body(body)

    def test_body_over_the_cap_raises(self):
        body = " ".join(["word"] * (MAX_SKILL_BODY_TOKENS + 1))
        with pytest.raises(SkillTooLarge):
            lint_skill_body(body)

    def test_custom_max_tokens_is_respected(self):
        with pytest.raises(SkillTooLarge):
            lint_skill_body("one two three four five", max_tokens=3)


class TestRenderFrameText:
    def test_includes_name_version_description_and_body(self):
        manifest, body = parse_skill_md(VALID_SKILL_MD)
        text = render_frame_text(manifest, body)

        assert "cmake.diagnose_configure" in text
        assert "v1" in text
        assert manifest.description in text
        assert "Diagnose Configure" in text
