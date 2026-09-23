"""Contract tests for the envelope schema (ftw_plan.md §4)."""

import pytest
from pydantic import ValidationError

from ftw.protocol import (
    SCHEMA_VERSION,
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
    MessageType,
    ProgressEnvelope,
    ProgressPayload,
    ResultEnvelope,
    ResultPayload,
    dump_envelope,
    parse_envelope,
)


def make_call() -> CallEnvelope:
    return CallEnvelope(
        source="repl.master",
        target="worker.tool.shell",
        deadline_ms=30_000,
        payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}),
    )


class TestDefaults:
    def test_schema_version_defaults(self):
        env = make_call()
        assert env.schema_version == SCHEMA_VERSION == 1

    def test_ids_are_auto_generated_and_unique(self):
        a, b = make_call(), make_call()
        assert a.msg_id and a.trace_id and a.span_id
        assert a.msg_id != b.msg_id
        assert a.trace_id != b.trace_id
        assert a.span_id != b.span_id

    def test_parent_span_id_defaults_to_none(self):
        assert make_call().parent_span_id is None

    def test_type_is_fixed_per_subclass(self):
        assert make_call().type == MessageType.CALL


class TestRoundTrip:
    @pytest.mark.parametrize(
        "envelope",
        [
            make_call(),
            ResultEnvelope(
                source="worker.tool.shell",
                target="repl.master",
                payload=ResultPayload(
                    status="ok",
                    summary="Ran echo",
                    outputs={"stdout": "hi\n"},
                    evidence=["exit code 0"],
                    cost={"tokens": 12, "wall_time_ms": 4},
                ),
            ),
            ErrorEnvelope(
                source="worker.tool.shell",
                target="repl.master",
                payload=ErrorPayload(code="deadline_exceeded", message="timed out"),
            ),
            AskEnvelope(
                source="worker.tool.shell",
                target="repl.master",
                payload=AskPayload(question="Overwrite file?", resume_token="tok-1"),
            ),
            AnswerEnvelope(
                source="repl.master",
                target="worker.tool.shell",
                payload=AnswerPayload(resume_token="tok-1", value=True),
            ),
            CancelEnvelope(
                source="repl.master",
                target="worker.tool.shell",
                payload=CancelPayload(target_span_id="abc-123", reason="user aborted"),
            ),
            ProgressEnvelope(
                source="worker.tool.shell",
                target="repl.master",
                payload=ProgressPayload(message="50% done", pct=0.5),
            ),
            EventEnvelope(
                source="worker.tool.shell",
                target="events",
                payload=EventPayload(topic="event.tool.shell.exit", data={"code": 0}),
            ),
            InterceptEnvelope(
                source="repl.master",
                target="repl.master",
                payload=InterceptPayload(decision="BLOCK", reason="path outside sandbox"),
            ),
        ],
    )
    def test_round_trips_through_json(self, envelope):
        raw = dump_envelope(envelope)
        assert isinstance(raw, bytes)
        restored = parse_envelope(raw)
        assert restored == envelope


class TestDiscriminatedParsing:
    def test_parse_envelope_dispatches_by_type(self):
        raw = dump_envelope(make_call())
        restored = parse_envelope(raw)
        assert isinstance(restored, CallEnvelope)
        assert restored.payload.action == "run_command"

    def test_parse_envelope_accepts_dict(self):
        env = make_call()
        as_dict = env.model_dump(mode="json")
        restored = parse_envelope(as_dict)
        assert restored == env

    def test_unknown_type_is_rejected(self):
        raw = dump_envelope(make_call()).decode()
        raw = raw.replace('"CALL"', '"BOGUS"')
        with pytest.raises(ValidationError):
            parse_envelope(raw)

    def test_wrong_payload_shape_is_rejected(self):
        with pytest.raises(ValidationError):
            CallEnvelope(
                source="a",
                target="b",
                payload=ResultPayload(status="ok"),  # type: ignore[arg-type]
            )


class TestSchemaExport:
    def test_export_json_schemas_writes_one_file_per_envelope_type(self, tmp_path):
        from ftw.protocol import export_json_schemas

        written = export_json_schemas(tmp_path)

        assert set(written) == {t.value for t in MessageType}
        for message_type, path in written.items():
            assert path.exists()
            assert path.name == f"{message_type.lower()}.schema.json"

    def test_exported_schema_is_valid_json_schema_shape(self, tmp_path):
        import json

        from ftw.protocol import export_json_schemas

        written = export_json_schemas(tmp_path)
        schema = json.loads(written["CALL"].read_text())
        assert schema["title"] == "CallEnvelope"
        assert "payload" in schema["properties"]


class TestTopicFraming:
    def test_frame_and_parse_event_round_trip(self):
        from ftw.protocol import frame_event, parse_event

        env = EventEnvelope(
            source="worker.tool.shell",
            target="events",
            payload=EventPayload(topic="event.tool.shell.exit", data={"code": 0}),
        )
        framed = frame_event(env)
        assert framed.startswith(b"event.tool.shell.exit\x00{")
        topic, restored = parse_event(framed)
        assert topic == "event.tool.shell.exit"
        assert restored == env
