"""Fuzz tests for skills/manifest.py's parse_skill_md - the other place
arbitrary text (a SKILL.md a user wrote, or copied from somewhere) flows
into a parser (YAML frontmatter + a regex split) before anything gets
close to a model. Two things worth automated adversarial coverage that
the example-based tests in test_skills_manifest.py don't attempt:

1. Crash resistance: parse_skill_md must never raise anything other than
   SkillParseError, on any input.
2. Timing: the frontmatter regex (_FRONTMATTER_PATTERN, DOTALL, a single
   non-greedy group) isn't shaped like the classic ReDoS pattern
   (nested quantifiers), but "isn't shaped like" is an argument, not a
   measurement - an explicit per-example deadline turns that argument
   into something Hypothesis actually checks on every generated example,
   including adversarial "almost-a-delimiter" dash patterns designed to
   maximize backtracking.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from ftw.skills.manifest import SkillParseError, parse_skill_md

# 1s per example is generous for a file this small (the real cap this
# guards against is catastrophic - exponential - blowup, not ordinary
# variance), while still being far under pytest's own 10s global timeout
# so a genuine regression fails with a clear per-example Hypothesis
# report instead of the whole test run just hanging.
_TIMING_SETTINGS = settings(max_examples=300, deadline=1000)


class TestCrashResistance:
    @given(st.text(max_size=2000))
    @_TIMING_SETTINGS
    def test_arbitrary_text_never_raises_anything_but_skillparseerror(self, text):
        try:
            parse_skill_md(text)
        except SkillParseError:
            pass

    @given(st.binary(max_size=2000))
    @_TIMING_SETTINGS
    def test_arbitrary_bytes_decoded_as_text_never_crash(self, data):
        # a worker/store always hands parse_skill_md a str (Path.read_text()
        # already decoded it) - this simulates that decoding step too,
        # rather than assuming utf-8 input is the only thing that reaches
        # parse_skill_md in practice.
        text = data.decode("utf-8", errors="replace")
        try:
            parse_skill_md(text)
        except SkillParseError:
            pass

    @given(frontmatter=st.text(max_size=300), body=st.text(max_size=300))
    @_TIMING_SETTINGS
    def test_frontmatter_shaped_garbage_never_crashes(self, frontmatter, body):
        """More likely than pure random text to actually reach the YAML
        parser (rather than failing the outer '---...---' regex match
        immediately) - this is what exercises yaml.safe_load's own
        error handling, not just _FRONTMATTER_PATTERN's."""
        text = f"---\n{frontmatter}\n---\n{body}"
        try:
            parse_skill_md(text)
        except SkillParseError:
            pass


class TestTimingUnderAdversarialInput:
    """Specifically targets _FRONTMATTER_PATTERN with inputs built to
    maximize near-misses on the '---' delimiter it's searching for -
    the shape of input that would trigger catastrophic backtracking in a
    vulnerable regex. Hypothesis's own per-example deadline (1s, see
    _TIMING_SETTINGS) is what actually catches a regression here, not
    the assertions in the test body."""

    @given(st.lists(st.sampled_from(["-", "--", "---", "\n", "a", " "]), max_size=400).map("".join))
    @_TIMING_SETTINGS
    def test_pathological_dash_patterns_stay_fast(self, text):
        try:
            parse_skill_md(text)
        except SkillParseError:
            pass

    @given(st.integers(min_value=0, max_value=500))
    @_TIMING_SETTINGS
    def test_many_near_miss_delimiters_stay_fast(self, n):
        # "---\n" repeated with no closing delimiter ever actually
        # completing the frontmatter block - the pattern's own non-greedy
        # group has to walk past all of them before failing.
        text = "---\n" + ("almost---\n" * n)
        try:
            parse_skill_md(text)
        except SkillParseError:
            pass
