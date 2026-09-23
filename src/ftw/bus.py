"""NNG socket wrappers: Req/Rep (with contexts), Pub/Sub, and lifecycle cleanup
(ftw_plan.md §4).

Design notes, matched to decisions made while planning:

* **REQ resend is disabled, not tuned.** NNG's ``Req0`` resends an unanswered
  request after ``resend_time`` (default 60s), which would execute a slow
  call twice. Rather than compute a resend time from each call's deadline,
  :class:`Requester` disables resend outright (``resend_time = -1``) and
  detects lateness itself via ``recv_timeout``, raising
  :class:`DeadlineExceeded`. This avoids the duplicate-execution hazard
  entirely instead of racing against it.
* **Idempotency is enforced by the callee.** :class:`Replier` deduplicates
  on ``idempotency_key`` so a caller that legitimately retries (e.g. after
  its own timeout) is safe to call again with the same key.
* **Concurrency comes from pynng contexts**, not one thread per connection:
  several worker threads each hold their own ``Context`` on the same
  ``Rep0`` socket, and nng dispatches inbound requests across them.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import pynng
from pydantic import ValidationError

from ftw.protocol import (
    AnyEnvelope,
    ErrorEnvelope,
    ErrorPayload,
    EventEnvelope,
    dump_envelope,
    frame_event,
    parse_envelope,
    parse_event,
)

Handler = Callable[[AnyEnvelope], AnyEnvelope]

_DEFAULT_TIMEOUT_MS = 30_000
_DEFAULT_POLL_MS = 100


class DeadlineExceeded(Exception):
    """Raised by :class:`Requester` when a reply doesn't arrive in time.

    This is the low-level, Pythonic signal for callers of the bus. Code that
    needs to hand a failure across the wire (e.g. a coordinator relaying a
    delegated call's outcome) converts it with :meth:`to_error_envelope`.
    """

    def to_error_envelope(self, *, source: str, target: str, **kwargs) -> ErrorEnvelope:
        return ErrorEnvelope(
            source=source,
            target=target,
            payload=ErrorPayload(code="deadline_exceeded", message=str(self) or "deadline exceeded"),
            **kwargs,
        )


def _socket_path_for_ipc(address: str) -> Path | None:
    if not address.startswith("ipc://"):
        return None
    return Path(address[len("ipc://") :])


class Requester:
    """A REQ socket dialed to one target. One call in flight at a time;
    open several instances (or one per thread) for concurrency."""

    def __init__(self, address: str, default_timeout_ms: int = _DEFAULT_TIMEOUT_MS):
        self._socket = pynng.Req0(dial=address)
        self._socket.resend_time = -1  # never let NNG silently re-send a call
        self._default_timeout_ms = default_timeout_ms

    @property
    def resend_time(self) -> int:
        return self._socket.resend_time

    def call(self, envelope: AnyEnvelope, timeout_ms: int | None = None) -> AnyEnvelope:
        # NOTE: a real Ctrl-C during this call is not reliably delivered.
        # pynng's blocking send()/recv() don't release the GIL, so no
        # thread — not even a background one running this same call — gets
        # a chance to process a pending signal until the call itself
        # returns (confirmed empirically). A short-poll retry loop was
        # tried and reverted: retrying recv() on the same Req0 socket after
        # a timeout hits a separate pynng bug (BadState) unless paired with
        # resend + idempotency-key-based retry, which is real, separate
        # work — see agent_loop.py's _send_cancel docstring.
        timeout = timeout_ms if timeout_ms is not None else (envelope.deadline_ms or self._default_timeout_ms)
        self._socket.send_timeout = timeout
        self._socket.recv_timeout = timeout
        try:
            self._socket.send(dump_envelope(envelope))
        except pynng.exceptions.Timeout as exc:
            raise DeadlineExceeded("timed out sending call") from exc
        try:
            raw = self._socket.recv()
        except pynng.exceptions.Timeout as exc:
            raise DeadlineExceeded("timed out waiting for reply") from exc
        return parse_envelope(raw)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "Requester":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _IdempotencyCache:
    """Thread-safe: a repeated key returns the cached reply or waits for the
    in-flight call to finish and returns its result — the handler runs at
    most once per key, and distinct keys never block one another."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done: dict[str, bytes] = {}
        self._inflight: dict[str, threading.Event] = {}

    def get_or_compute(self, key: str, compute: Callable[[], bytes]) -> bytes:
        with self._lock:
            if key in self._done:
                return self._done[key]
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait()
            with self._lock:
                if key in self._done:
                    return self._done[key]
                raise RuntimeError(f"idempotent call for key {key!r} failed in the owning thread")

        try:
            result = compute()
        except BaseException:
            with self._lock:
                del self._inflight[key]
            event.set()
            raise

        with self._lock:
            self._done[key] = result
            del self._inflight[key]
        event.set()
        return result


class Replier:
    """A REP socket serving concurrent requests via ``num_workers`` pynng
    contexts. Binds at construction; for ``ipc://`` addresses, a stale
    socket file left by a crashed run is removed and the bind retried."""

    def __init__(self, address: str, num_workers: int = 4, recv_timeout_ms: int = _DEFAULT_POLL_MS):
        self._address = address
        self._num_workers = num_workers
        self._dedupe = _IdempotencyCache()
        self._socket = self._bind(address, recv_timeout_ms)

    @staticmethod
    def _bind(address: str, recv_timeout_ms: int) -> pynng.Rep0:
        try:
            return pynng.Rep0(listen=address, recv_timeout=recv_timeout_ms)
        except pynng.exceptions.AddressInUse:
            stale_path = _socket_path_for_ipc(address)
            if stale_path is None or not stale_path.exists():
                raise
            stale_path.unlink()
            return pynng.Rep0(listen=address, recv_timeout=recv_timeout_ms)

    @staticmethod
    def _error_reply(envelope: AnyEnvelope, exc: Exception) -> ErrorEnvelope:
        return ErrorEnvelope(
            source=envelope.target,
            target=envelope.source,
            trace_id=envelope.trace_id,
            payload=ErrorPayload(code="handler_error", message=f"{type(exc).__name__}: {exc}"),
        )

    def _handle(self, envelope: AnyEnvelope, handler: Handler) -> bytes:
        key = envelope.idempotency_key
        if key is None:
            return dump_envelope(handler(envelope))
        return self._dedupe.get_or_compute(key, lambda: dump_envelope(handler(envelope)))

    def _worker_loop(self, handler: Handler, stop_event: threading.Event | None) -> None:
        ctx = self._socket.new_context()
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    raw = ctx.recv()
                except pynng.exceptions.Timeout:
                    continue
                except pynng.exceptions.Closed:
                    return
                try:
                    envelope = parse_envelope(raw)
                except ValidationError:
                    continue  # malformed on the wire; drop rather than crash the worker
                try:
                    reply_bytes = self._handle(envelope, handler)
                except Exception as exc:
                    # A handler bug (or an upstream failure it didn't catch
                    # — a ProviderError, a UnicodeDecodeError on binary
                    # subprocess output, ...) must not take this whole
                    # worker thread down with it: that's num_workers
                    # concurrent slots reduced by one, permanently, for
                    # the life of the process. Reply with an error instead
                    # and keep serving.
                    reply_bytes = dump_envelope(self._error_reply(envelope, exc))
                try:
                    ctx.send(reply_bytes)
                except pynng.exceptions.Closed:
                    return
        finally:
            try:
                ctx.close()
            except pynng.exceptions.Closed:
                pass

    def serve_forever(self, handler: Handler, stop_event: threading.Event | None = None) -> None:
        """Blocks until ``stop_event`` is set (or forever if omitted).
        Run this in its own thread."""
        threads = [
            threading.Thread(target=self._worker_loop, args=(handler, stop_event), daemon=True)
            for _ in range(self._num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def close(self) -> None:
        self._socket.close()
        stale_path = _socket_path_for_ipc(self._address)
        if stale_path is not None and stale_path.exists():
            stale_path.unlink()

    def __enter__(self) -> "Replier":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class DispatchRouter:
    """Routes an envelope to the Requester bound to its ``target``.

    A caller like AgentLoop takes a single ``dispatch`` callable, but more
    than one worker (the shell worker, the delegated skill runner, ...)
    can be in play in the same session, each bound to its own address.
    This is what lets one callable still reach the right one, keyed by the
    envelope's logical target name rather than a raw address — the same
    role ``tool_targets`` plays in labeling a CallEnvelope, just followed
    through to an actual socket."""

    def __init__(self, requesters: dict[str, Requester]):
        self._requesters = requesters

    def __call__(self, envelope: AnyEnvelope) -> AnyEnvelope:
        requester = self._requesters.get(envelope.target)
        if requester is None:
            raise KeyError(f"no requester configured for target {envelope.target!r} (known: {sorted(self._requesters)})")
        return requester.call(envelope)


class Publisher:
    """PUB socket. Sends are fire-and-forget: with no subscriber connected,
    NNG discards the message immediately rather than blocking."""

    def __init__(self, address: str):
        self._socket = pynng.Pub0(listen=address)

    def publish(self, envelope: EventEnvelope) -> None:
        self._socket.send(frame_event(envelope))

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "Publisher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Subscriber:
    """SUB socket pre-subscribed to a set of topic prefixes (byte-prefix
    match against the ``<topic>\\0<json>`` framing in protocol.py)."""

    def __init__(self, address: str, topics: list[str], recv_timeout_ms: int = 1000):
        self._socket = pynng.Sub0(dial=address, recv_timeout=recv_timeout_ms)
        for topic in topics:
            self._socket.subscribe(topic.encode("utf-8"))

    def recv(self, timeout_ms: int | None = None) -> EventEnvelope:
        if timeout_ms is not None:
            self._socket.recv_timeout = timeout_ms
        try:
            raw = self._socket.recv()
        except pynng.exceptions.Timeout as exc:
            raise DeadlineExceeded("timed out waiting for event") from exc
        _topic, envelope = parse_event(raw)
        return envelope

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "Subscriber":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
