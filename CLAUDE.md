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
killing the worker thread, made `AgentLoop` catch `ProviderError` and
`DeadlineExceeded` gracefully instead of crashing the turn, gave every
dispatched `CALL` a deadline longer than whatever it wraps
(`DEFAULT_CALL_DEADLINES_MS`), fixed `ftw tap`'s default address to
actually match the REPL's, and switched turn/pin frame-tagging from
per-turn to per-message (`ContextWorkbench.add_tagged_turn`) so the
mount/unmount invariant below holds even when a model mounts, works, and
unmounts within a single reply. Phase 4 (interceptor policy beyond
"confirm every shell command", deterministic verifiers) is next. See §8 of
`ftw_plan.md` for the phase list.

**Known gap:** Ctrl-C does not reliably trigger that CANCEL. Confirmed
empirically (see `tests/test_bus.py::TestRecvIsInterruptible`, marked
`xfail`): pynng's blocking `send()`/`recv()` don't release the GIL, so no
thread — not even a background one running the same call — gets a chance
to process a pending SIGINT until the call itself returns. A short-poll
retry loop was tried and reverted: retrying `recv()` on the same `Req0`
socket after a timeout hits a separate pynng bug (`BadState`). A real fix
needs resend- and idempotency-key-based retry in `bus.Requester.call()`
(`resend_time` is disabled outright today, for the correctness reasons in
that class's own docstring) — real, separate work, not a quick patch. If a
`KeyboardInterrupt` or a `DeadlineExceeded` *does* land (e.g. a dispatch
that was already returning), `AgentLoop` now reports it as a normal tool
result instead of crashing the turn — but the underlying "Ctrl-C usually
doesn't get a chance to land at all" problem above is unchanged.

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
  `asyncio.Queue`, direct function calls, or other stand-ins "for now".
  Tests use `inproc://`; runtime uses `ipc://`.
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
- Deps: `pynng`, `pydantic`, `pyyaml`, `httpx`; dev: `pytest`, `pytest-timeout`.
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
