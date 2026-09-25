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
  ``num_workers`` asyncio Tasks, all on the ONE event loop of the owning
  process, each hold their own ``Context`` on the same ``Rep0`` socket, and
  nng dispatches inbound requests across them.
* **Everything here is async, on purpose, to fix a real bug**: pynng's
  blocking ``send()``/``recv()`` don't release control back to the
  interpreter in time for a real SIGINT to be processed, so Ctrl-C during a
  long call couldn't reliably reach the CANCEL machinery
  (see ``tests/test_bus.py::TestRecvIsInterruptible`` and
  ``CLAUDE.md``'s Known Gap). ``asend()``/``arecv()`` use NNG's real
  ``nng_aio``/``nng_aio_cancel`` C API via pynng's asyncio integration —
  confirmed empirically to interrupt a blocked wait promptly and to leave
  the socket safely reusable afterward. ``Publisher``/``Subscriber`` are the
  one deliberate exception: they're only used by the standalone ``ftw tap``
  diagnostic viewer, not anywhere in the Ctrl-C-affected interactive path,
  so they stay synchronous — not an oversight, a scope call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Self

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

Handler = Callable[[AnyEnvelope], Awaitable[AnyEnvelope]]

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

    async def call(self, envelope: AnyEnvelope, timeout_ms: int | None = None) -> AnyEnvelope:
        # asend()/arecv() use NNG's real nng_aio/nng_aio_cancel API (via
        # pynng's asyncio integration), not a blocking C call — a real
        # Ctrl-C, wired up as loop.add_signal_handler(SIGINT, task.cancel)
        # by the caller (see repl/session.py), reliably interrupts a call
        # in progress here: asyncio.CancelledError propagates out of
        # whichever of asend()/arecv() was in flight, and the socket is
        # safely reusable for a later call afterward (confirmed
        # empirically). This is the actual fix for the gap the old
        # blocking-recv docstring here used to describe.
        timeout = timeout_ms if timeout_ms is not None else (envelope.deadline_ms or self._default_timeout_ms)
        self._socket.send_timeout = timeout
        self._socket.recv_timeout = timeout
        try:
            await self._socket.asend(dump_envelope(envelope))
        except pynng.exceptions.Timeout as exc:
            raise DeadlineExceeded("timed out sending call") from exc
        try:
            raw = await self._socket.arecv()
        except pynng.exceptions.Timeout as exc:
            raise DeadlineExceeded("timed out waiting for reply") from exc
        return parse_envelope(raw)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _IdempotencyCache:
    """A repeated key returns the cached reply or waits for the in-flight
    call to finish and returns its result — the handler runs at most once
    per key, and distinct keys never block one another.

    No lock: every worker Task for a given Replier runs on that Replier's
    one event loop, and the check-then-mutate dict spans below (get/set,
    with no ``await`` in between) can't be interleaved by another Task —
    a context switch only happens *at* an ``await`` point. This is a real
    simplification versus the old threading.Lock version, not just a
    rename — but it depends on that one-event-loop invariant holding. Don't
    spread a Replier's worker Tasks across threads or event loops."""

    def __init__(self) -> None:
        self._done: dict[str, bytes] = {}
        self._inflight: dict[str, asyncio.Event] = {}

    async def get_or_compute(self, key: str, compute: Callable[[], Awaitable[bytes]]) -> bytes:
        if key in self._done:
            return self._done[key]
        existing = self._inflight.get(key)
        if existing is not None:
            await existing.wait()
            if key in self._done:
                return self._done[key]
            raise RuntimeError(f"idempotent call for key {key!r} failed in the owning task")

        event = asyncio.Event()
        self._inflight[key] = event
        try:
            result = await compute()
        except BaseException:
            del self._inflight[key]
            event.set()
            raise

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

    async def _handle(self, envelope: AnyEnvelope, handler: Handler) -> bytes:
        key = envelope.idempotency_key
        if key is None:
            return dump_envelope(await handler(envelope))

        async def compute() -> bytes:
            return dump_envelope(await handler(envelope))

        return await self._dedupe.get_or_compute(key, compute)

    async def _worker_loop(self, handler: Handler, stop_event: asyncio.Event | None) -> None:
        ctx = self._socket.new_context()
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    raw = await ctx.arecv()
                except pynng.exceptions.Timeout:
                    continue
                except pynng.exceptions.Closed:
                    return
                try:
                    envelope = parse_envelope(raw)
                except ValidationError:
                    continue  # malformed on the wire; drop rather than crash the worker
                try:
                    reply_bytes = await self._handle(envelope, handler)
                except Exception as exc:  # noqa: BLE001 - see the comment below for why this must stay blind
                    # A handler bug (or an upstream failure it didn't catch
                    # — a ProviderError, a UnicodeDecodeError on binary
                    # subprocess output, ...) must not take this whole
                    # worker Task down with it: that's num_workers
                    # concurrent slots reduced by one, permanently, for
                    # the life of the process. Reply with an error instead
                    # and keep serving. asyncio.CancelledError is a
                    # BaseException, not caught here — it must propagate.
                    reply_bytes = dump_envelope(self._error_reply(envelope, exc))
                try:
                    await ctx.asend(reply_bytes)
                except pynng.exceptions.Closed:
                    return
        finally:
            try:
                ctx.close()
            except pynng.exceptions.Closed:
                pass

    async def serve_forever(self, handler: Handler, stop_event: asyncio.Event | None = None) -> None:
        """Blocks until ``stop_event`` is set (or forever if omitted).
        Await this as its own Task alongside whatever else runs on the
        same event loop."""
        async with asyncio.TaskGroup() as tg:
            for _ in range(self._num_workers):
                tg.create_task(self._worker_loop(handler, stop_event))

    def close(self) -> None:
        self._socket.close()
        stale_path = _socket_path_for_ipc(self._address)
        if stale_path is not None and stale_path.exists():
            stale_path.unlink()

    def __enter__(self) -> Self:
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

    async def __call__(self, envelope: AnyEnvelope) -> AnyEnvelope:
        requester = self._requesters.get(envelope.target)
        if requester is None:
            raise KeyError(f"no requester configured for target {envelope.target!r} (known: {sorted(self._requesters)})")
        return await requester.call(envelope)


class Publisher:
    """PUB socket. Sends are fire-and-forget: with no subscriber connected,
    NNG discards the message immediately rather than blocking."""

    def __init__(self, address: str):
        self._socket = pynng.Pub0(listen=address)

    def publish(self, envelope: EventEnvelope) -> None:
        self._socket.send(frame_event(envelope))

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class Subscriber:
    """SUB socket pre-subscribed to a set of topic prefixes (byte-prefix
    match against the ``<topic>\\0<json>`` framing in protocol.py).

    Deliberately stays synchronous (unlike Requester/Replier above): its
    blocking recv() is structurally the same shape as Requester.call()'s
    old blocking recv and *could* move to arecv() the same way later with
    little new design work, but it's only used by the standalone `ftw tap`
    diagnostic viewer, not anywhere in the Ctrl-C-affected interactive
    path — migrating it now would add surface area for zero bug-fix value."""

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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
