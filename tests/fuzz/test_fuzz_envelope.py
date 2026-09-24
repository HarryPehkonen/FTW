"""Fuzz/property tests for protocol.py's parse_envelope/dump_envelope -
the real wire boundary. Highest-value fuzz target in the project:
Replier._worker_loop specifically catches pydantic.ValidationError when
parsing an incoming envelope (bus.py) - anything parse_envelope raises
OTHER than ValidationError would propagate past that except clause and
take a worker Task down. These tests exist to catch that class of
regression, not a specific known bug.

Opt-in only - see CLAUDE.md's Testing section for how to run this
directory and pyproject.toml's `fuzz` dependency group / norecursedirs
entry for how the isolation works.
"""

import pydantic
from hypothesis import given, settings
from hypothesis import strategies as st

from ftw.protocol import (
    AnswerEnvelope,
    AnswerPayload,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    CallPayload,
    CancelEnvelope,
    CancelPayload,
    ErrorEnvelope,
    ErrorPayload,
    EventEnvelope,
    EventPayload,
    InterceptEnvelope,
    InterceptPayload,
    ProgressEnvelope,
    ProgressPayload,
    ResultEnvelope,
    ResultPayload,
    dump_envelope,
    parse_envelope,
)

# A recursive JSON-compatible value, for the Any-typed fields (CallPayload
# .args, AnswerPayload.value, ResultPayload.outputs/cost, ErrorPayload
# .detail, EventPayload.data, InterceptPayload.modified_payload). NaN/inf
# excluded: json round-tripping them is a red herring here (NaN != NaN
# breaks the round-trip equality assertion for reasons unrelated to the
# code under test), not something parse_envelope needs to defend against
# differently from any other float.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=50),
)
_json_value = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=20), children, max_size=5),
    ),
    max_leaves=10,
)
_json_dict = st.dictionaries(st.text(max_size=20), _json_value, max_size=5)

_short_text = st.text(max_size=50)
_nonempty_text = st.text(min_size=1, max_size=50)
_common = {"source": _nonempty_text, "target": _nonempty_text}

_envelope_strategy = st.one_of(
    st.builds(CallEnvelope, payload=st.builds(CallPayload, action=_nonempty_text, args=_json_dict), **_common),
    st.builds(
        ResultEnvelope,
        payload=st.builds(
            ResultPayload,
            status=st.sampled_from(["ok", "error"]),
            summary=_short_text,
            outputs=_json_dict,
            evidence=st.lists(_short_text, max_size=5),
            cost=_json_dict,
        ),
        **_common,
    ),
    st.builds(ErrorEnvelope, payload=st.builds(ErrorPayload, code=_nonempty_text, message=_short_text, detail=_json_dict), **_common),
    st.builds(
        AskEnvelope,
        payload=st.builds(AskPayload, question=_short_text, resume_token=_nonempty_text, expected=_json_dict),
        **_common,
    ),
    st.builds(AnswerEnvelope, payload=st.builds(AnswerPayload, resume_token=_nonempty_text, value=_json_value), **_common),
    st.builds(CancelEnvelope, payload=st.builds(CancelPayload, target_span_id=_nonempty_text, reason=_short_text), **_common),
    st.builds(
        ProgressEnvelope,
        payload=st.builds(ProgressPayload, message=_short_text, pct=st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))),
        **_common,
    ),
    st.builds(EventEnvelope, payload=st.builds(EventPayload, topic=_short_text, data=_json_dict), **_common),
    st.builds(
        InterceptEnvelope,
        payload=st.builds(
            InterceptPayload,
            decision=st.sampled_from(["ALLOW", "MODIFY", "BLOCK", "ASK"]),
            reason=_short_text,
            modified_payload=st.one_of(st.none(), _json_dict),
        ),
        **_common,
    ),
)


class TestRoundTrip:
    """A stronger correctness property than "doesn't crash": every valid
    envelope, across every message type, survives dump -> parse with
    every field intact - not just the handful of hand-picked field
    combinations the example-based tests in test_protocol.py happen to
    cover."""

    @given(_envelope_strategy)
    @settings(max_examples=300, deadline=None)
    def test_dump_then_parse_round_trips_exactly(self, envelope):
        reparsed = parse_envelope(dump_envelope(envelope))
        assert reparsed == envelope


class TestCrashResistance:
    """The actual real-world risk: bus.Replier._worker_loop catches
    pydantic.ValidationError specifically when parsing bytes off the
    wire (see bus.py's own comment there). Anything parse_envelope
    raises instead of ValidationError - a KeyError, a RecursionError, a
    plain crash - would propagate past that except clause and take the
    worker Task down. These generate structurally arbitrary input (not
    just noise) and assert only ValidationError (or success) ever comes
    back out."""

    @given(st.binary(max_size=500))
    @settings(max_examples=300, deadline=None)
    def test_arbitrary_bytes_never_raise_anything_but_validationerror(self, data):
        try:
            parse_envelope(data)
        except pydantic.ValidationError:
            pass

    @given(st.text(max_size=500))
    @settings(max_examples=300, deadline=None)
    def test_arbitrary_text_never_raises_anything_but_validationerror(self, text):
        try:
            parse_envelope(text)
        except pydantic.ValidationError:
            pass

    @given(_json_value)
    @settings(max_examples=300, deadline=None)
    def test_arbitrary_json_compatible_python_values_never_raise_anything_but_validationerror(self, value):
        try:
            parse_envelope(value)
        except pydantic.ValidationError:
            pass

    @given(st.dictionaries(st.text(max_size=20), _json_value, max_size=10))
    @settings(max_examples=300, deadline=None)
    def test_dict_shaped_garbage_never_raises_anything_but_validationerror(self, data):
        """More likely than pure random JSON to land close to a real
        envelope's shape (a `type` key with a garbage value, a `payload`
        key with the wrong shape, ...) - the discriminated-union
        resolution path is exactly where a stray KeyError/AttributeError
        would be most likely to hide."""
        try:
            parse_envelope(data)
        except pydantic.ValidationError:
            pass
