"""Contract tests for the NNG socket wrappers (ftw_plan.md §4).

Covers: inproc round trip, deadline timeouts, duplicate-call suppression via
idempotency_key, an ASK/ANSWER round trip, REQ resend being disabled, real
Ctrl-C interruption of a blocked call, and a small ipc:// integration test
including stale-socket cleanup.

Requester/Replier/DispatchRouter are async (asend()/arecv(), NNG's real
nng_aio/nng_aio_cancel API) so a real SIGINT can actually interrupt a call
in progress - see bus.py's module docstring. Publisher/Subscriber stay
synchronous on purpose (see Subscriber's own docstring) - their tests are
plain `def`, not `async def`.
"""

import asyncio
import os
import signal
import threading
import time

import pytest

from ftw.bus import (
    DeadlineExceeded,
    DispatchRouter,
    Publisher,
    Replier,
    Requester,
    Subscriber,
)
from ftw.protocol import (
    AnswerEnvelope,
    AnswerPayload,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    CallPayload,
    ErrorEnvelope,
    EventEnvelope,
    EventPayload,
    ResultEnvelope,
    ResultPayload,
)
from ftw.runtime import inproc_address, ipc_address, runtime_dir


def make_call(**kwargs) -> CallEnvelope:
    return CallEnvelope(
        source="repl.master",
        target="worker.tool.shell",
        payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}),
        **kwargs,
    )


async def echo_handler(env: CallEnvelope) -> ResultEnvelope:
    return ResultEnvelope(
        source=env.target,
        target=env.source,
        parent_span_id=env.span_id,
        payload=ResultPayload(status="ok", summary="handled"),
    )


def start_server(replier: Replier, handler, stop_event: asyncio.Event) -> asyncio.Task:
    """Schedules replier.serve_forever(handler, stop_event) as a background
    Task on the test's own event loop - Replier's worker Tasks and the
    test's own Requester calls interleave at their respective await
    points, same event loop throughout (no threads). Callers should
    `stop_event.set()` then `await` the returned Task once done, so the
    Task is never left dangling when the test function returns."""
    return asyncio.ensure_future(replier.serve_forever(handler, stop_event))


class TestRequestReplyRoundTrip:
    async def test_call_and_reply_over_inproc(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        with Replier(addr) as rep:
            task = start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = await req.call(make_call())
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.summary == "handled"
            assert reply.parent_span_id is not None
            stop.set()
        await task


class TestResendDisabled:
    def test_requester_disables_req_resend(self, unique_name):
        addr = inproc_address(unique_name)
        with Requester(addr) as req:
            assert req.resend_time == -1


class TestDeadlineHandling:
    async def test_slow_handler_raises_deadline_exceeded(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        async def slow_handler(env):
            await asyncio.sleep(0.5)
            return await echo_handler(env)

        with Replier(addr) as rep:
            task = start_server(rep, slow_handler, stop)
            with Requester(addr) as req, pytest.raises(DeadlineExceeded):
                await req.call(make_call(), timeout_ms=100)
            stop.set()
        await task

    async def test_deadline_exceeded_converts_to_error_envelope(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        async def slow_handler(env):
            await asyncio.sleep(0.5)
            return await echo_handler(env)

        with Replier(addr) as rep:
            task = start_server(rep, slow_handler, stop)
            with Requester(addr) as req:
                call = make_call()
                try:
                    await req.call(call, timeout_ms=100)
                    pytest.fail("expected DeadlineExceeded")
                except DeadlineExceeded as exc:
                    err = exc.to_error_envelope(source=call.target, target=call.source)
            assert err.payload.code == "deadline_exceeded"
            stop.set()
        await task


class TestRecvIsInterruptible:
    """A real Ctrl-C now reliably interrupts a blocked call: asend()/arecv()
    use NNG's real nng_aio/nng_aio_cancel API (confirmed via a throwaway
    script this session, 5/5 runs), not the old blocking send()/recv()
    that swallowed a pending SIGINT until the call itself returned. This
    class used to hold one xfail test documenting that gap - it's fixed
    for real now that Requester.call() is async, wired up the way
    repl/session.py's per-turn Task cancellation does it."""

    async def test_a_real_sigint_interrupts_a_long_wait_promptly(self, unique_name):
        addr = inproc_address(unique_name)
        # No Replier bound: with nothing ever accepting the dial, asend()
        # blocks for the full send_timeout below (confirmed empirically) -
        # the "hung worker" shape this test needs, no never-replying
        # handler/worker Task to manage during teardown.
        loop = asyncio.get_running_loop()
        with Requester(addr) as req:
            task = asyncio.ensure_future(req.call(make_call(), timeout_ms=2000))
            loop.add_signal_handler(signal.SIGINT, task.cancel)

            def send_sigint_soon():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=send_sigint_soon, daemon=True).start()

            start = time.monotonic()
            with pytest.raises(asyncio.CancelledError):
                await task
            elapsed = time.monotonic() - start
            loop.remove_signal_handler(signal.SIGINT)

        assert elapsed < 1.0  # interrupted well before the 2s deadline

    async def test_the_socket_is_still_usable_after_a_cancelled_call(self, unique_name):
        """A cancelled call must not corrupt the Req0 socket - the next
        normal call on the same Requester still succeeds (confirmed
        empirically this session). This is what lets repl/session.py
        reuse one Requester/DispatchRouter for the life of the REPL
        session instead of reconnecting after every Ctrl-C."""
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def handler(env):
            calls.append(env)
            if len(calls) == 1:
                await asyncio.sleep(0.3)  # long enough to cancel well before it would reply
            return await echo_handler(env)

        with Replier(addr) as rep:
            task = start_server(rep, handler, stop)
            with Requester(addr) as req:
                first = asyncio.ensure_future(req.call(make_call(), timeout_ms=2000))
                await asyncio.sleep(0.05)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first

                reply = await req.call(make_call(), timeout_ms=2000)
                assert reply.payload.summary == "handled"
            stop.set()
        await task


class TestIdempotencyDedupe:
    async def test_duplicate_idempotency_key_only_invokes_handler_once(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def counting_handler(env):
            calls.append(env.msg_id)
            return ResultEnvelope(
                source=env.target,
                target=env.source,
                payload=ResultPayload(status="ok", summary=f"invocation {len(calls)}"),
            )

        with Replier(addr) as rep:
            task = start_server(rep, counting_handler, stop)
            with Requester(addr) as req:
                call = make_call(idempotency_key="fixed-key-123")
                first = await req.call(call)
                second = await req.call(call)
            assert len(calls) == 1
            assert first.payload.summary == second.payload.summary == "invocation 1"
            stop.set()
        await task

    async def test_different_idempotency_keys_both_invoke_handler(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def counting_handler(env):
            calls.append(env.msg_id)
            return await echo_handler(env)

        with Replier(addr) as rep:
            task = start_server(rep, counting_handler, stop)
            with Requester(addr) as req:
                await req.call(make_call(idempotency_key="key-a"))
                await req.call(make_call(idempotency_key="key-b"))
            assert len(calls) == 2
            stop.set()
        await task

    async def test_concurrent_duplicate_key_calls_still_invoke_handler_once(self, unique_name):
        """The old threading.Lock-based cache was proven safe under real
        concurrent duplicate calls from separate OS threads; the new
        lock-free (single-event-loop) version needs the same guarantee
        under concurrent asyncio Tasks instead."""
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def counting_handler(env):
            calls.append(env.msg_id)
            await asyncio.sleep(0.05)  # widen the race window
            return ResultEnvelope(
                source=env.target,
                target=env.source,
                payload=ResultPayload(status="ok", summary=f"invocation {len(calls)}"),
            )

        with Replier(addr, num_workers=4) as rep:
            task = start_server(rep, counting_handler, stop)

            async def caller():
                with Requester(addr) as req:
                    return await req.call(make_call(idempotency_key="shared-key"), timeout_ms=2000)

            results = await asyncio.gather(caller(), caller(), caller())
            assert len(calls) == 1
            assert all(r.payload.summary == "invocation 1" for r in results)
            stop.set()
        await task


class TestAskAnswerRoundTrip:
    async def test_handler_can_suspend_on_ask_and_resume_on_answer(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        pending: dict[str, CallEnvelope] = {}

        async def stateful_handler(env):
            if isinstance(env, CallEnvelope):
                token = "resume-tok-1"
                pending[token] = env
                return AskEnvelope(
                    source=env.target,
                    target=env.source,
                    parent_span_id=env.span_id,
                    payload=AskPayload(question="proceed?", resume_token=token),
                )
            if isinstance(env, AnswerEnvelope):
                original = pending.pop(env.payload.resume_token)
                assert env.payload.value is True
                return ResultEnvelope(
                    source=original.target,
                    target=original.source,
                    parent_span_id=original.span_id,
                    payload=ResultPayload(status="ok", summary="resumed and completed"),
                )
            raise AssertionError(f"unexpected envelope type: {type(env)}")

        with Replier(addr) as rep:
            task = start_server(rep, stateful_handler, stop)
            with Requester(addr) as req:
                asked = await req.call(make_call())
                assert isinstance(asked, AskEnvelope)
                assert asked.payload.question == "proceed?"

                answer = AnswerEnvelope(
                    source="repl.master",
                    target="worker.tool.shell",
                    payload=AnswerPayload(resume_token=asked.payload.resume_token, value=True),
                )
                final = await req.call(answer)
            assert isinstance(final, ResultEnvelope)
            assert final.payload.summary == "resumed and completed"
            stop.set()
        await task


class TestConcurrentContexts:
    async def test_replier_serves_overlapping_calls_concurrently(self, unique_name):
        """Three slow calls in flight at once must all complete well under
        their combined latency, proving separate contexts (now: separate
        asyncio Tasks on one event loop, not separate OS threads) serve
        them rather than one call blocking the others. A synchronous
        (non-awaiting) handler here would freeze the whole loop for every
        Task, not just its own - that's why this handler awaits
        asyncio.sleep, never time.sleep."""
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        async def slow_handler(env):
            await asyncio.sleep(0.2)
            return await echo_handler(env)

        with Replier(addr, num_workers=4) as rep:
            task = start_server(rep, slow_handler, stop)

            async def caller():
                with Requester(addr) as req:
                    return await req.call(make_call(), timeout_ms=2000)

            t0 = time.monotonic()
            results = await asyncio.gather(caller(), caller(), caller())
            elapsed = time.monotonic() - t0

            assert len(results) == 3
            # Sequential would take >= 0.6s; concurrent contexts should finish well under that.
            assert elapsed < 0.5
            stop.set()
        await task


class TestIpcTransport:
    async def test_round_trip_over_ipc(self, isolated_runtime_dir):
        addr = ipc_address("test-ipc-worker")
        stop = asyncio.Event()
        with Replier(addr) as rep:
            task = start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = await req.call(make_call())
            assert reply.payload.summary == "handled"
            stop.set()
        await task

    async def test_stale_socket_file_is_removed_before_bind(self, isolated_runtime_dir):
        addr = ipc_address("stale-worker")
        path = runtime_dir() / "stale-worker.ipc"
        path.touch()  # dangling file left by a crashed run; no listener behind it

        stop = asyncio.Event()
        with Replier(addr) as rep:
            task = start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = await req.call(make_call())
            assert reply.payload.summary == "handled"
            stop.set()
        await task

    def test_socket_file_is_removed_on_close(self, isolated_runtime_dir):
        addr = ipc_address("cleanup-worker")
        path = runtime_dir() / "cleanup-worker.ipc"
        with Replier(addr):
            assert path.exists()
        assert not path.exists()


class TestHandlerExceptionSafety:
    """A handler raising anything other than the pynng-level exceptions
    _worker_loop already expects (Timeout, Closed) must not kill that
    worker Task outright - the caller gets a clean ErrorEnvelope, and
    later calls still get served by the same, still-alive worker pool."""

    async def test_handler_exception_becomes_an_error_envelope(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        async def flaky_handler(env):
            raise ValueError("boom")

        with Replier(addr) as rep:
            task = start_server(rep, flaky_handler, stop)
            with Requester(addr) as req:
                reply = await req.call(make_call(), timeout_ms=2000)
            assert isinstance(reply, ErrorEnvelope)
            assert reply.payload.code == "handler_error"
            assert "boom" in reply.payload.message
            stop.set()
        await task

    async def test_worker_survives_and_serves_the_next_call(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def handler(env):
            calls.append(env)
            if len(calls) == 1:
                raise ValueError("first call blows up")
            return await echo_handler(env)

        with Replier(addr, num_workers=1) as rep:  # force the second call onto the SAME worker
            task = start_server(rep, handler, stop)
            with Requester(addr) as req:
                first = await req.call(make_call(), timeout_ms=2000)
                assert isinstance(first, ErrorEnvelope)
                second = await req.call(make_call(), timeout_ms=2000)
                assert isinstance(second, ResultEnvelope)
            stop.set()
        await task

    async def test_exception_from_a_deduped_idempotent_call_still_replies_cleanly(self, unique_name):
        addr = inproc_address(unique_name)
        stop = asyncio.Event()

        async def flaky_handler(env):
            raise RuntimeError("always fails")

        with Replier(addr) as rep:
            task = start_server(rep, flaky_handler, stop)
            with Requester(addr) as req:
                reply = await req.call(make_call(idempotency_key="k1"), timeout_ms=2000)
            assert isinstance(reply, ErrorEnvelope)
            stop.set()
        await task

    async def test_a_cancelled_worker_does_not_take_the_whole_replier_down(self, unique_name):
        """asyncio.CancelledError propagating out of a handler (the caller
        gave up, e.g. via a real Ctrl-C reaching a downstream call) must
        not be swallowed the way an ordinary handler exception is -
        _worker_loop's except clause is `except Exception`, never `except
        BaseException`, so the one worker Task handling that call ends
        for good (its context is still cleanly closed via `finally`).
        What must survive is the REPLIER as a whole: TaskGroup only
        cancels sibling tasks when a child fails with something other
        than CancelledError (see the stdlib docs), so a second worker
        stays alive and keeps serving - num_workers=1 would give this
        test no sibling to prove that with, so it uses 2."""
        addr = inproc_address(unique_name)
        stop = asyncio.Event()
        calls = []

        async def handler(env):
            calls.append(env)
            if len(calls) == 1:
                raise asyncio.CancelledError("simulating a cancelled downstream call")
            return await echo_handler(env)

        with Replier(addr, num_workers=2) as rep:
            task = start_server(rep, handler, stop)
            with Requester(addr) as req:
                with pytest.raises(DeadlineExceeded):
                    # the one worker that picked this up dies mid-request
                    # (CancelledError propagates past _worker_loop entirely,
                    # per the docstring above) so this call never gets a
                    # reply - expected; what matters is the call below.
                    await req.call(make_call(), timeout_ms=300)
                second = await req.call(make_call(), timeout_ms=2000)
                assert isinstance(second, ResultEnvelope)
            stop.set()
        await task


class TestDispatchRouter:
    """AgentLoop's `dispatch` contract is a single async callable - but more
    than one worker (the shell worker, the delegated skill runner, ...)
    can be in play at once, each bound to its own address. DispatchRouter
    is what lets one `dispatch` callable still reach the right one, keyed
    by the envelope's logical `target` name rather than a raw address."""

    async def test_routes_by_envelope_target_to_the_matching_requester(self, unique_name):
        addr_a, addr_b = inproc_address(f"{unique_name}-a"), inproc_address(f"{unique_name}-b")
        stop = asyncio.Event()

        async def handler_a(env):
            return await echo_handler(env)

        async def handler_b(env):
            return ResultEnvelope(source="worker.b", target=env.source, payload=ResultPayload(status="ok", summary="from b"))

        call_a = CallEnvelope(source="repl.master", target="worker.a", payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}))
        call_b = CallEnvelope(source="repl.master", target="worker.b", payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}))

        with Replier(addr_a) as rep_a, Replier(addr_b) as rep_b:
            task_a = start_server(rep_a, handler_a, stop)
            task_b = start_server(rep_b, handler_b, stop)
            with Requester(addr_a) as req_a, Requester(addr_b) as req_b:
                router = DispatchRouter({"worker.a": req_a, "worker.b": req_b})

                reply_a = await router(call_a)
                reply_b = await router(call_b)

            assert reply_a.payload.summary == "handled"
            assert reply_b.payload.summary == "from b"
            stop.set()
        await task_a
        await task_b

    async def test_unknown_target_raises_a_clear_error(self):
        router = DispatchRouter({})
        call = CallEnvelope(source="repl.master", target="worker.nope", payload=CallPayload(action="run_command", args={}))
        with pytest.raises(KeyError, match="worker.nope"):
            await router(call)


class TestPubSub:
    """Publisher/Subscriber stay synchronous - see Subscriber's own
    docstring in bus.py for why. Plain `def` tests, not `async def`."""

    def make_event(self, topic: str) -> EventEnvelope:
        return EventEnvelope(
            source="worker.tool.shell",
            target="events",
            payload=EventPayload(topic=topic, data={"code": 0}),
        )

    def test_publish_with_no_subscriber_does_not_block(self, unique_name):
        addr = inproc_address(unique_name)
        with Publisher(addr) as pub:
            t0 = time.time()
            pub.publish(self.make_event("event.tool.shell.exit"))
            assert time.time() - t0 < 0.5

    def test_subscriber_receives_matching_topic(self, unique_name):
        addr = inproc_address(unique_name)
        with Publisher(addr) as pub, Subscriber(addr, topics=["event.tool."]) as sub:
            time.sleep(0.05)  # let the subscription establish
            env = self.make_event("event.tool.shell.exit")
            pub.publish(env)
            received = sub.recv(timeout_ms=1000)
            assert received == env

    def test_subscriber_does_not_receive_nonmatching_topic(self, unique_name):
        addr = inproc_address(unique_name)
        with Publisher(addr) as pub, Subscriber(addr, topics=["event.tool."]) as sub:
            time.sleep(0.05)
            pub.publish(self.make_event("event.memory.write"))
            with pytest.raises(DeadlineExceeded):
                sub.recv(timeout_ms=200)
