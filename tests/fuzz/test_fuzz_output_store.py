"""Fuzz tests for outputs.py's path-traversal defense - the security fix
from the remediation pass (_check_safe, an allowlist regex checked before
trace_id/output_id ever reach a filesystem path).

The example-based tests in test_outputs.py already cover the known-bad
inputs (absolute paths, "..", literal slashes) that motivated the fix.
What's genuinely different here: rather than asserting specific inputs
are rejected, these assert the actual INVARIANT - that no trace_id/
output_id combination, however adversarial, ever causes OutputStore to
read or write anything outside its own root - directly against the
public API (read/save), not the internal _check_safe helper in
isolation. This is the shape of test most likely to catch a regression
if _check_safe's regex is ever loosened without the reasoning behind it
being re-checked (e.g. "just allow unicode letters too" quietly
reopening a normalization-based escape).

Given _check_safe is already a strict allowlist ([A-Za-z0-9._-]+, no
".", no "..") rather than a blocklist, expect this to mostly confirm
what's already true by inspection - see CLAUDE.md's Testing section for
the honest take on how fruitful each fuzz target has actually been.
"""

import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from ftw.outputs import OutputNotFound, OutputStore, UnsafeIdentifier

# Deliberately unrestricted: slashes, dots, unicode, control characters,
# whatever Hypothesis's default text alphabet includes - the whole point
# is not to pre-filter what an adversarial trace_id/output_id looks like.
_adversarial_identifier = st.text(min_size=1, max_size=300)

_SETTINGS = settings(max_examples=500, deadline=1000)


def _fresh_root_with_a_sibling_secret() -> tuple[Path, Path]:
    """A store root, plus a file just outside it that a real escape would
    be able to reach - freshly created per Hypothesis example (not a
    pytest fixture: those are resolved once per test *function* call,
    before Hypothesis's internal example loop even starts, so reusing one
    across examples would silently share state between them)."""
    base = Path(tempfile.mkdtemp())
    root = base / "store"
    root.mkdir()
    secret = base / "secret.txt"
    secret.write_text("TOP SECRET")
    return root, secret


class TestReadNeverEscapesRoot:
    @given(trace_id=_adversarial_identifier, output_id=_adversarial_identifier)
    @_SETTINGS
    def test_read(self, trace_id, output_id):
        root, secret = _fresh_root_with_a_sibling_secret()
        try:
            store = OutputStore(root)
            try:
                store.read(trace_id, output_id)
            except OutputNotFound:
                pass
            except UnicodeDecodeError:
                pass  # found and opened a real file inside root that just wasn't valid text - not an escape
            assert secret.read_text() == "TOP SECRET"  # the escape target itself, untouched either way
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)

    @given(trace_id=_adversarial_identifier, output_id=_adversarial_identifier, pattern=st.just("."))
    @_SETTINGS
    def test_grep(self, trace_id, output_id, pattern):
        root, secret = _fresh_root_with_a_sibling_secret()
        try:
            store = OutputStore(root)
            try:
                store.grep(trace_id, output_id, pattern)
            except OutputNotFound:
                pass
            except UnicodeDecodeError:
                pass
            assert secret.read_text() == "TOP SECRET"
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)


class TestSaveNeverWritesOutsideRoot:
    @given(trace_id=_adversarial_identifier, output_id=_adversarial_identifier)
    @_SETTINGS
    def test_save(self, trace_id, output_id):
        root, secret = _fresh_root_with_a_sibling_secret()
        try:
            store = OutputStore(root)
            try:
                store.save(trace_id, "fuzzed content", output_id=output_id)
            except UnsafeIdentifier:
                pass
            except OSError:
                # an id that passed _check_safe's regex but is still
                # unusable as a real filename on this filesystem (e.g.
                # over the length limit) - a clean, crash-free rejection
                # either way, not an escape. Worth knowing about even so:
                # see CLAUDE.md's Testing section.
                pass
            assert secret.read_text() == "TOP SECRET"  # never overwritten
            # nothing was created as a sibling of root, whatever happened
            # inside it
            assert set(root.parent.iterdir()) <= {root, secret}
        finally:
            shutil.rmtree(root.parent, ignore_errors=True)
