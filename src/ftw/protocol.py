"""Envelope, payload, and error models — the NNG wire protocol (ftw_plan.md §4).

Every message that crosses the bus is one of the ``MessageType`` variants
below, serialized as a strict JSON envelope. ``parse_envelope``/``dump_envelope``
are the only sanctioned way on/off the wire; nothing is ever pickled.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = 1


class MessageType(str, Enum):
    CALL = "CALL"
    RESULT = "RESULT"
    ERROR = "ERROR"
    ASK = "ASK"
    ANSWER = "ANSWER"
    CANCEL = "CANCEL"
    PROGRESS = "PROGRESS"
    EVENT = "EVENT"
    INTERCEPT = "INTERCEPT"


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CallPayload(_Payload):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class ResultPayload(_Payload):
    status: Literal["ok", "error"] = "ok"
    summary: str = ""
    outputs: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    cost: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(_Payload):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AskPayload(_Payload):
    question: str
    resume_token: str
    expected: dict[str, Any] = Field(default_factory=dict)


class AnswerPayload(_Payload):
    resume_token: str
    value: Any = None


class CancelPayload(_Payload):
    target_span_id: str
    reason: str = ""


class ProgressPayload(_Payload):
    message: str = ""
    pct: float | None = None


class EventPayload(_Payload):
    topic: str
    data: dict[str, Any] = Field(default_factory=dict)


class InterceptPayload(_Payload):
    decision: Literal["ALLOW", "MODIFY", "BLOCK", "ASK"]
    reason: str = ""
    modified_payload: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Envelopes
# --------------------------------------------------------------------------


class _EnvelopeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    msg_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    source: str
    target: str
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    brief: str | None = None


class CallEnvelope(_EnvelopeBase):
    type: Literal[MessageType.CALL] = MessageType.CALL
    payload: CallPayload


class ResultEnvelope(_EnvelopeBase):
    type: Literal[MessageType.RESULT] = MessageType.RESULT
    payload: ResultPayload


class ErrorEnvelope(_EnvelopeBase):
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    payload: ErrorPayload


class AskEnvelope(_EnvelopeBase):
    type: Literal[MessageType.ASK] = MessageType.ASK
    payload: AskPayload


class AnswerEnvelope(_EnvelopeBase):
    type: Literal[MessageType.ANSWER] = MessageType.ANSWER
    payload: AnswerPayload


class CancelEnvelope(_EnvelopeBase):
    type: Literal[MessageType.CANCEL] = MessageType.CANCEL
    payload: CancelPayload


class ProgressEnvelope(_EnvelopeBase):
    type: Literal[MessageType.PROGRESS] = MessageType.PROGRESS
    payload: ProgressPayload


class EventEnvelope(_EnvelopeBase):
    type: Literal[MessageType.EVENT] = MessageType.EVENT
    payload: EventPayload


class InterceptEnvelope(_EnvelopeBase):
    type: Literal[MessageType.INTERCEPT] = MessageType.INTERCEPT
    payload: InterceptPayload


AnyEnvelope = Annotated[
    Union[
        CallEnvelope,
        ResultEnvelope,
        ErrorEnvelope,
        AskEnvelope,
        AnswerEnvelope,
        CancelEnvelope,
        ProgressEnvelope,
        EventEnvelope,
        InterceptEnvelope,
    ],
    Field(discriminator="type"),
]

_envelope_adapter: TypeAdapter[AnyEnvelope] = TypeAdapter(AnyEnvelope)

_ENVELOPE_CLASSES_BY_TYPE: dict[str, type[BaseModel]] = {
    MessageType.CALL: CallEnvelope,
    MessageType.RESULT: ResultEnvelope,
    MessageType.ERROR: ErrorEnvelope,
    MessageType.ASK: AskEnvelope,
    MessageType.ANSWER: AnswerEnvelope,
    MessageType.CANCEL: CancelEnvelope,
    MessageType.PROGRESS: ProgressEnvelope,
    MessageType.EVENT: EventEnvelope,
    MessageType.INTERCEPT: InterceptEnvelope,
}


def export_json_schemas(directory: str | Path) -> dict[str, Path]:
    """Write one JSON Schema file per envelope type to ``directory``, for
    polyglot workers (§4, §7) that aren't using Pydantic to validate the wire
    format. Returns a map of message-type value to the file written."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for message_type, cls in _ENVELOPE_CLASSES_BY_TYPE.items():
        schema = cls.model_json_schema()
        path = out_dir / f"{message_type.lower()}.schema.json"
        path.write_text(json.dumps(schema, indent=2) + "\n")
        written[message_type.value] = path
    return written


def parse_envelope(data: bytes | str | dict[str, Any]) -> AnyEnvelope:
    """Parse raw wire data into the correctly-typed envelope subclass."""
    if isinstance(data, (bytes, str)):
        return _envelope_adapter.validate_json(data)
    return _envelope_adapter.validate_python(data)


def dump_envelope(envelope: AnyEnvelope) -> bytes:
    """Serialize an envelope to the canonical JSON wire form."""
    return envelope.model_dump_json().encode("utf-8")


# --------------------------------------------------------------------------
# PUB/SUB topic framing (ftw_plan.md §4: "<topic>\0<json>")
# --------------------------------------------------------------------------


def frame_event(envelope: EventEnvelope) -> bytes:
    """Frame an EVENT envelope for PUB/SUB so SUB's byte-prefix topic
    filter can match on the topic instead of every message's leading ``{``.
    """
    return envelope.payload.topic.encode("utf-8") + b"\x00" + dump_envelope(envelope)


def parse_event(raw: bytes) -> tuple[str, EventEnvelope]:
    """Inverse of ``frame_event``."""
    topic_bytes, _, body = raw.partition(b"\x00")
    envelope = parse_envelope(body)
    if not isinstance(envelope, EventEnvelope):
        raise ValueError(f"framed message is not an EVENT envelope: {envelope.type}")
    return topic_bytes.decode("utf-8"), envelope
