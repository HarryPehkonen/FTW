# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

FTW (Framed Task Workers) is a local-first agent framework: a REPL whose LLM
context is an explicitly budgeted "workbench", skills that can be **mounted**
into that context and **unmounted** to free the space, and workers that talk
over an NNG message bus. The design lives in `ftw_plan.md`; read it before
making architectural changes.

**Status:** Phase 0 through Phase 3 are done, including delegated skill
runs (`delegate_skill`, driven by `AgentLoop.run_delegated` +
`skills/runner.py`): a skill runs as an isolated sub-task in its own
subprocess with a fresh, isolated workbench, the caller's context grows
only by its structured `Result`, a delegated run's own interceptor ASK
suspends and relays back through the caller to a real human instead of
blocking (`AgentLoop.resume_delegated`, `DelegatedSuspension`), and
`CANCEL` on a separate control channel (`<address>.ctl`) cooperatively
aborts an in-flight run between steps — all fully built and tested via the
control channel directly. A follow-up hardening pass (post-Phase-3, driven
by an external review) fixed a path-traversal hole in `outputs.py`, made
`bus.Replier` turn a handler exception into an `ErrorEnvelope` instead of
killing the worker (thread, at the time), made `AgentLoop` catch
`ProviderError` and `DeadlineExceeded` gracefully instead of crashing the
turn, gave every dispatched `CALL` a deadline longer than whatever it
wraps (`DEFAULT_CALL_DEADLINES_MS`), fixed `ftw tap`'s default address to
actually match the REPL's, and switched turn/pin frame-tagging from
per-turn to per-message (`ContextWorkbench.add_tagged_turn`) so the
mount/unmount invariant below holds even when a model mounts, works, and
unmounts within a single reply.

A second follow-up pass then took the whole call chain — `bus.Requester`/
`bus.Replier`/`bus.DispatchRouter`, `AgentLoop`, `skills/runner.py`,
`repl/session.py`/`repl/cli.py`, the provider adapters, and the shell
worker — from synchronous/threaded to `asyncio`, specifically to fix the
Ctrl-C gap described in the (now historical) Known Gap this replaces:
pynng's `arecv()`/`asend()` use NNG's real `nng_aio`/`nng_aio_cancel` C
API, so a real SIGINT (wired up in `repl/session.py` as
`loop.add_signal_handler(SIGINT, turn_task.cancel)`) now reliably
interrupts a blocked bus call, confirmed by
`tests/test_bus.py::TestRecvIsInterruptible`, no longer `xfail`.
`AgentLoop._step_loop` is a plain coroutine, not an async generator —
PEP 525 forbids `return <value>` in one, which the old synchronous
generator's suspend/resume design relied on — so a delegated run's
suspend/resume (`DelegatedSuspension`) is now built on an `asyncio.Task`
plus a small internal `_Asker` (`asyncio.Future`-based), not a paused
generator. `Replier`'s `num_workers` are now concurrent `asyncio.Task`s on
one event loop per process, not OS threads; `_IdempotencyCache` and
`skills/runner.py`'s pending-ask/cancel-flag tracking dropped their
`threading.Lock`s entirely (not replaced with `asyncio.Lock`) since a
single event loop makes the check-then-mutate dict spans involved atomic
by construction — see the comments on those classes for the one invariant
that depends on. Provider HTTP calls (`httpx.AsyncClient`) and shell
subprocess calls (`asyncio.create_subprocess_exec`) became genuinely async
I/O too, not just thread-bridged, so a hung LLM call or shell command is
now cancellable mid-flight, not just abandoned in the background.

Phase 4 (interceptor policy beyond "confirm every shell command",
deterministic verifiers) is next. See §8 of `ftw_plan.md` for the phase
list.

**Known gap:** Ctrl-C at an idle REPL prompt (no turn in flight) isn't
reliably cancellable. `repl/session.py`'s line reading is bridged through
`asyncio.to_thread(input_fn, prompt)` rather than rewritten as a native
async reader, specifically to keep `input()`'s automatic GNU-readline
integration (arrow-key history/editing); the trade-off is that cancelling
the *awaiting* Task unblocks the asyncio side, but the underlying OS
thread stays blocked on the real stdin read until something is actually
typed or EOF arrives. This is narrower and separate from the bug the async
migration above fixed (a blocked *bus call during a turn*, which is fully
solved — `repl/session.py`'s `_run_turn_interruptible` cancels that
promptly via a real SIGINT). Ctrl-C during an actual turn works; Ctrl-C
while just sitting at the prompt with nothing running does not yet.

**Known gap:** `idempotency_key` (`protocol.py`, `bus.py`'s
`_IdempotencyCache`) is real, tested dedupe plumbing, but nothing in this
codebase sets it on an outgoing `CallEnvelope` yet — no caller actually
retries a call today, so the dedupe path is exercised only by tests that
construct the key by hand. Treat "workers dedupe on idempotency_key" below
as a capability, not a claim that it's protecting anything in production
yet.

Run it: `uv run ftw` (needs a real key for the `fast`/`smart` tiers in
`ftw.toml` — `DEEPSEEK_API_KEY` / `NOUS_API_KEY`). Point `--skills-dir` at
`examples/skills` to try mounting right away, e.g.
`uv run ftw --skills-dir examples/skills`. `uv run ftw tap` streams live
events from a running session in another terminal.

## Non-negotiables

- **NNG from day one.** Every component boundary goes over `pynng`. Don't use
  `asyncio.Queue`, direct function calls, or other stand-ins "for now" —
  this is about crossing a component boundary specifically; using
  `asyncio` itself for concurrency *within* a process (the bus, the agent
  loop, and the REPL are all `async` now) is the norm, not an exception to
  this rule. Tests use `inproc://`; runtime uses `ipc://`.
- **Mount/unmount is the headline feature.** Mounted skills form a frame tree
  (multiple siblings and nesting allowed); the model mounts/unmounts itself and
  the user can do it manually. Unmounting a frame must return the workbench to
  its pre-mount footprint plus a small milestone
  (`after_unmount == before_mount + milestone`). Any leak is a bug.
- **Sockets live in `$XDG_RUNTIME_DIR/ftw/` (0700), never `/tmp`.** Workers
  dedupe on `idempotency_key`; the bus controls REQ `resend_time`.
- **API keys come from the environment only** (`DEEPSEEK_API_KEY`,
  `NOUS_API_KEY`). Never write them to config, traces, or events.
- **Strict TDD, fully offline tests.** No network, no live LLM tokens in the
  test suite. Use the scripted mock provider and `inproc://` transport.
  Write the failing test first.
- **JSON envelopes on the wire.** The envelope schema is the contract between
  languages; Python-only shortcuts (pickle, passing objects) are not allowed
  across the bus.
- **Files are the source of truth.** Skills, memory, and traces are plain
  Markdown/YAML/JSONL under `$FTW_HOME`. Indexes must be rebuildable from them.
- **Skills are task-scoped and ≤ 1,500 tokens** (body), measured with the
  project's canonical token counter, not a model-specific one.

## Environment

- Python 3.14 (system). `pynng` 0.9.0 ships a cp314 manylinux wheel.
- **uv** manages the project (no pip/Poetry): `uv sync`, `uv run pytest`,
  `uv run ftw`. Add deps with `uv add`.
- Deps: `pynng`, `pydantic`, `pyyaml`, `httpx`; dev: `pytest`, `pytest-timeout`,
  `pytest-asyncio` (`asyncio_mode = "auto"` in `pyproject.toml` — every
  `async def test_...` just works, no `@pytest.mark.asyncio` needed).
- Providers: DeepSeek and Nous Research Portal, both via the OpenAI-compatible
  adapter (see §6 of the plan). Live-provider checks are opt-in smoke tests,
  never part of `pytest`.
- Regenerate `schemas/*.schema.json` after touching `protocol.py`:
  `uv run scripts/export_schemas.py`.

## Testing

`uv run pytest` (10s per-test timeout is configured; a hang is a bug, not
something to wait out). All bus tests run over `inproc://`; ipc:// tests use
the `isolated_runtime_dir` fixture so they never touch the real runtime dir.
Every module lands with its test file written and shown failing *first* —
don't skip the red step even when the implementation feels obvious.

## Layout

See §7 of `ftw_plan.md`; phases are in §8. Source goes in `src/ftw/`, tests in `tests/`.

## Git

Sole contributor. Commit and push directly to `main`; no pull requests.
Ask before force-pushing or rewriting history.
