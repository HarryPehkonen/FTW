"""Contract tests for the NNG socket wrappers (ftw_plan.md §4).

Covers: inproc round trip, deadline timeouts, duplicate-call suppression via
idempotency_key, an ASK/ANSWER round trip, REQ resend being disabled, and a
small ipc:// integration test including stale-socket cleanup.
"""

import os
import signal
import threading
import time

import pytest

from ftw.bus import DeadlineExceeded, Publisher, Replier, Requester, Subscriber
from ftw.protocol import (
    AnswerEnvelope,
    AnswerPayload,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    CallPayload,
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


def echo_handler(env: CallEnvelope) -> ResultEnvelope:
    return ResultEnvelope(
        source=env.target,
        target=env.source,
        parent_span_id=env.span_id,
        payload=ResultPayload(status="ok", summary="handled"),
    )


def start_server(replier: Replier, handler, stop_event: threading.Event) -> threading.Thread:
    thread = threading.Thread(target=replier.serve_forever, args=(handler, stop_event), daemon=True)
    thread.start()
    return thread


class TestRequestReplyRoundTrip:
    def test_call_and_reply_over_inproc(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()
        with Replier(addr) as rep:
            start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = req.call(make_call())
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.summary == "handled"
            assert reply.parent_span_id is not None
            stop.set()


class TestResendDisabled:
    def test_requester_disables_req_resend(self, unique_name):
        addr = inproc_address(unique_name)
        with Requester(addr) as req:
            assert req.resend_time == -1


class TestDeadlineHandling:
    def test_slow_handler_raises_deadline_exceeded(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()

        def slow_handler(env):
            time.sleep(0.5)
            return echo_handler(env)

        with Replier(addr) as rep:
            start_server(rep, slow_handler, stop)
            with Requester(addr) as req:
                with pytest.raises(DeadlineExceeded):
                    req.call(make_call(), timeout_ms=100)
            stop.set()

    def test_deadline_exceeded_converts_to_error_envelope(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()

        def slow_handler(env):
            time.sleep(0.5)
            return echo_handler(env)

        with Replier(addr) as rep:
            start_server(rep, slow_handler, stop)
            with Requester(addr) as req:
                call = make_call()
                try:
                    req.call(call, timeout_ms=100)
                    pytest.fail("expected DeadlineExceeded")
                except DeadlineExceeded as exc:
                    err = exc.to_error_envelope(source=call.target, target=call.source)
            assert err.payload.code == "deadline_exceeded"
            stop.set()


class TestRecvIsInterruptible:
    """A single long blocking recv() can't be interrupted by a real
    Ctrl-C: pynng's send()/recv() don't release the GIL, so no thread —
    not even a background one running the same call — gets a chance to
    process a pending signal until the call itself returns. Confirmed
    empirically; a naive short-poll retry loop was tried and reverted
    because retrying recv() on the same Req0 socket after a timeout hits
    a separate pynng bug (BadState). Fixing this for real needs resend +
    idempotency-key-based retry (bus.py's Requester already disables
    resend outright for the correctness reasons in its own docstring), so
    it's real, separate work, not a quick patch alongside this one. This
    test documents the gap and will start passing on its own once that
    lands — see Requester.call()'s docstring."""

    @pytest.mark.xfail(reason="pynng's recv() doesn't release the GIL; see class docstring", strict=True)
    def test_a_real_sigint_interrupts_a_long_wait_promptly(self, unique_name):
        addr = inproc_address(unique_name)
        # No Replier bound — send() succeeds (inproc dial always "succeeds"
        # even with nothing listening; see bus.py's docstring), but no
        # reply will ever arrive, so recv() would otherwise block for the
        # full timeout below.
        with Requester(addr) as req:

            def send_sigint_soon():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=send_sigint_soon, daemon=True).start()

            start = time.monotonic()
            with pytest.raises(KeyboardInterrupt):
                req.call(make_call(), timeout_ms=2000)
            elapsed = time.monotonic() - start

        assert elapsed < 1.0  # interrupted well before the 2s deadline


class TestIdempotencyDedupe:
    def test_duplicate_idempotency_key_only_invokes_handler_once(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()
        calls = []

        def counting_handler(env):
            calls.append(env.msg_id)
            return ResultEnvelope(
                source=env.target,
                target=env.source,
                payload=ResultPayload(status="ok", summary=f"invocation {len(calls)}"),
            )

        with Replier(addr) as rep:
            start_server(rep, counting_handler, stop)
            with Requester(addr) as req:
                call = make_call(idempotency_key="fixed-key-123")
                first = req.call(call)
                second = req.call(call)
            assert len(calls) == 1
            assert first.payload.summary == second.payload.summary == "invocation 1"
            stop.set()

    def test_different_idempotency_keys_both_invoke_handler(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()
        calls = []

        def counting_handler(env):
            calls.append(env.msg_id)
            return echo_handler(env)

        with Replier(addr) as rep:
            start_server(rep, counting_handler, stop)
            with Requester(addr) as req:
                req.call(make_call(idempotency_key="key-a"))
                req.call(make_call(idempotency_key="key-b"))
            assert len(calls) == 2
            stop.set()


class TestAskAnswerRoundTrip:
    def test_handler_can_suspend_on_ask_and_resume_on_answer(self, unique_name):
        addr = inproc_address(unique_name)
        stop = threading.Event()
        pending: dict[str, CallEnvelope] = {}

        def stateful_handler(env):
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
            start_server(rep, stateful_handler, stop)
            with Requester(addr) as req:
                asked = req.call(make_call())
                assert isinstance(asked, AskEnvelope)
                assert asked.payload.question == "proceed?"

                answer = AnswerEnvelope(
                    source="repl.master",
                    target="worker.tool.shell",
                    payload=AnswerPayload(resume_token=asked.payload.resume_token, value=True),
                )
                final = req.call(answer)
            assert isinstance(final, ResultEnvelope)
            assert final.payload.summary == "resumed and completed"
            stop.set()


class TestConcurrentContexts:
    def test_replier_serves_overlapping_calls_concurrently(self, unique_name):
        """Two slow calls in flight at once must both complete well under
        their combined latency, proving separate contexts serve them rather
        than one call blocking the other."""
        addr = inproc_address(unique_name)
        stop = threading.Event()

        def slow_handler(env):
            time.sleep(0.2)
            return echo_handler(env)

        with Replier(addr, num_workers=4) as rep:
            start_server(rep, slow_handler, stop)

            results = {}

            def caller(key):
                with Requester(addr) as req:
                    results[key] = req.call(make_call(), timeout_ms=2000)

            t0 = time.time()
            threads = [threading.Thread(target=caller, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            elapsed = time.time() - t0

            assert len(results) == 3
            # Sequential would take >= 0.6s; concurrent contexts should finish well under that.
            assert elapsed < 0.5
            stop.set()


class TestIpcTransport:
    def test_round_trip_over_ipc(self, isolated_runtime_dir):
        addr = ipc_address("test-ipc-worker")
        stop = threading.Event()
        with Replier(addr) as rep:
            start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = req.call(make_call())
            assert reply.payload.summary == "handled"
            stop.set()

    def test_stale_socket_file_is_removed_before_bind(self, isolated_runtime_dir):
        addr = ipc_address("stale-worker")
        path = runtime_dir() / "stale-worker.ipc"
        path.touch()  # dangling file left by a crashed run; no listener behind it

        stop = threading.Event()
        with Replier(addr) as rep:
            start_server(rep, echo_handler, stop)
            with Requester(addr) as req:
                reply = req.call(make_call())
            assert reply.payload.summary == "handled"
            stop.set()

    def test_socket_file_is_removed_on_close(self, isolated_runtime_dir):
        addr = ipc_address("cleanup-worker")
        path = runtime_dir() / "cleanup-worker.ipc"
        with Replier(addr):
            assert path.exists()
        assert not path.exists()


class TestDispatchRouter:
    """AgentLoop's `dispatch` contract is a single callable — but more than
    one worker (the shell worker, the skill runner, ...) can be in play at
    once, each bound to its own address. DispatchRouter is what lets one
    `dispatch` callable still reach the right one, keyed by the envelope's
    logical `target` name rather than a raw address."""

    def test_routes_by_envelope_target_to_the_matching_requester(self, unique_name):
        addr_a, addr_b = inproc_address(f"{unique_name}-a"), inproc_address(f"{unique_name}-b")
        stop = threading.Event()

        def handler_a(env):
            return echo_handler(env)

        def handler_b(env):
            return ResultEnvelope(source="worker.b", target=env.source, payload=ResultPayload(status="ok", summary="from b"))

        call_a = CallEnvelope(source="repl.master", target="worker.a", payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}))
        call_b = CallEnvelope(source="repl.master", target="worker.b", payload=CallPayload(action="run_command", args={"argv": ["echo", "hi"]}))

        with Replier(addr_a) as rep_a, Replier(addr_b) as rep_b:
            start_server(rep_a, handler_a, stop)
            start_server(rep_b, handler_b, stop)
            with Requester(addr_a) as req_a, Requester(addr_b) as req_b:
                from ftw.bus import DispatchRouter

                router = DispatchRouter({"worker.a": req_a, "worker.b": req_b})

                reply_a = router(call_a)
                reply_b = router(call_b)

            assert reply_a.payload.summary == "handled"
            assert reply_b.payload.summary == "from b"
            stop.set()

    def test_unknown_target_raises_a_clear_error(self, unique_name):
        from ftw.bus import DispatchRouter

        router = DispatchRouter({})
        call = CallEnvelope(source="repl.master", target="worker.nope", payload=CallPayload(action="run_command", args={}))
        with pytest.raises(KeyError, match="worker.nope"):
            router(call)


class TestPubSub:
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
