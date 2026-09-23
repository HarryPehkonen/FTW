"""SkillStore: loads a directory of SKILL.md files and answers find_skill
queries (ftw_plan.md §3.2 "the whole catalog never has to be in context").
"""

import pytest

from ftw.skills.manifest import SkillParseError, SkillTooLarge
from ftw.skills.registry import SkillNotFound, SkillStore

CMAKE_SKILL = """---
name: cmake.diagnose_configure
description: Diagnose failing CMake configuration and isolate root causes with evidence.
---

Reproduce the failure, grep the error log, report cause and fix.
"""

TOOLCHAIN_SKILL = """---
name: toolchain.verify_installed
description: Verify a compiler toolchain is installed and on PATH.
---

Check for the compiler binary and report its version.
"""

GIT_SKILL = """---
name: git.bisect
description: Binary-search commit history to find which commit introduced a regression.
---

Run git bisect start, mark good/bad, narrow down the offending commit.
"""


def write_skill(root, relpath: str, text: str) -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestLoadAndGet:
    def test_get_returns_manifest_and_body_for_a_loaded_skill(self, tmp_path):
        write_skill(tmp_path, "cmake/diagnose_configure", CMAKE_SKILL)
        store = SkillStore(tmp_path)

        manifest, body = store.get("cmake.diagnose_configure")

        assert manifest.name == "cmake.diagnose_configure"
        assert "Reproduce the failure" in body

    def test_get_unknown_skill_raises(self, tmp_path):
        store = SkillStore(tmp_path)
        with pytest.raises(SkillNotFound):
            store.get("nope.nothing")

    def test_load_all_finds_skills_at_any_depth(self, tmp_path):
        write_skill(tmp_path, "cmake/diagnose_configure", CMAKE_SKILL)
        write_skill(tmp_path, "toolchain/verify_installed", TOOLCHAIN_SKILL)

        store = SkillStore(tmp_path)

        assert set(store.load_all()) == {"cmake.diagnose_configure", "toolchain.verify_installed"}

    def test_malformed_skill_raises_with_path_context(self, tmp_path):
        write_skill(tmp_path, "broken/skill", "not even frontmatter")
        store = SkillStore(tmp_path)
        with pytest.raises(SkillParseError):
            store.load_all()

    def test_oversized_skill_body_raises(self, tmp_path):
        huge_body = " ".join(["word"] * 2000)
        write_skill(
            tmp_path,
            "huge/skill",
            f"---\nname: huge.skill\ndescription: too big\n---\n\n{huge_body}\n",
        )
        store = SkillStore(tmp_path)
        with pytest.raises(SkillTooLarge):
            store.load_all()

    def test_empty_store_directory_is_fine(self, tmp_path):
        store = SkillStore(tmp_path)
        assert store.load_all() == {}


class TestFindSkill:
    def make_store(self, tmp_path) -> SkillStore:
        write_skill(tmp_path, "cmake/diagnose_configure", CMAKE_SKILL)
        write_skill(tmp_path, "toolchain/verify_installed", TOOLCHAIN_SKILL)
        write_skill(tmp_path, "git/bisect", GIT_SKILL)
        return SkillStore(tmp_path)

    def test_finds_best_matching_skill_by_description(self, tmp_path):
        store = self.make_store(tmp_path)
        results = store.find("cmake configuration failing")
        assert results[0] == "cmake.diagnose_configure"

    def test_finds_by_domain_word_in_name(self, tmp_path):
        store = self.make_store(tmp_path)
        results = store.find("git commit regression bisect")
        assert results[0] == "git.bisect"

    def test_respects_top_k(self, tmp_path):
        store = self.make_store(tmp_path)
        assert len(store.find("compiler toolchain configuration commit", top_k=2)) <= 2

    def test_no_match_returns_empty_list_not_error(self, tmp_path):
        store = SkillStore(tmp_path)  # empty store
        assert store.find("anything") == []
