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

A third follow-up pass then closed the one remaining Ctrl-C case the
async migration didn't cover: an idle REPL prompt (no turn in flight)
used to hang completely on a real SIGINT — not "not cancellable", a full,
silent hang requiring the process to be killed from another terminal.
Root cause, confirmed by reading `asyncio/runners.py` directly rather
than assumed: `asyncio.run()`'s own `Runner` installs a two-stage SIGINT
handler — the *first* Ctrl-C only requests cooperative cancellation of
the top-level task, which has nowhere to land while that task is blocked
in a plain synchronous call rather than an `await` (a *second* Ctrl-C is
what actually raises `KeyboardInterrupt`, by design). The REPL's line
reading used to be bridged through `asyncio.to_thread`, which made this
worse, not better: cancelling the wrapping Task doesn't stop the
underlying OS thread once it's already running a blocking read, so it
stays orphaned — and `asyncio.run()`'s own shutdown
(`shutdown_default_executor()`) later deadlocks joining it. Fixed by
`repl/session.py`'s `_blocking_read`: reads now run directly on the main
thread (nothing else needs the event loop while blocked on a line of
input, whether at the idle prompt or answering a mid-turn confirmation),
with Python's plain default SIGINT handler temporarily restored for the
duration — a single real Ctrl-C now raises `KeyboardInterrupt`
immediately, exactly like an ordinary synchronous script, no orphaned
thread, no two-stage dance. Verified against the real `uv run ftw`
binary, not just unit tests: a real, correctly-targeted `SIGINT` (not
sent to `uv`'s own wrapper process, which is a distinct child-having
process, not an exec-replacement — confirmed empirically, another false
positive avoided) unblocks a genuinely-idle prompt in ~0.1s, repeatedly.
Answering a mid-turn confirmation prompt with a real Ctrl-C declines that
action *and*, in practice, ends the turn shortly after too (not "decline
only" as first documented — `loop.add_signal_handler`'s
`signal.set_wakeup_fd()` plumbing stays live underneath the temporary
plain-handler swap, so the same signal also reaches
`_run_turn_interruptible`'s own turn-level cancellation once
`_blocking_read` returns control at the next real `await`; confirmed by
reading `asyncio.unix_events` — a safe, tested outcome, just broader than
originally intended, not worth the real coupling a narrower fix would
need). Ctrl-C now works the same way everywhere in the REPL: during a
turn, during a confirmation prompt, and at an idle prompt.

**Known gap:** `idempotency_key` (`protocol.py`, `bus.py`'s
`_IdempotencyCache`) is real, tested dedupe plumbing, but nothing in this
codebase sets it on an outgoing `CallEnvelope` yet — no caller actually
retries a call today, so the dedupe path is exercised only by tests that
construct the key by hand. Treat "workers dedupe on idempotency_key" below
as a capability, not a claim that it's protecting anything in production
yet.

Run it: `uv run ftw` (needs a real key for the `fast`/`smart` tiers in
`ftw.toml` — `DEEPSEEK_API_KEY` / `NOUS_API_KEY`). `examples/skills` is
FTW's own bundled, read-only skill catalog (`repl/cli.py`'s
`bundled_skills_dir`) — always searched alongside whatever `--skills-dir`
points at, no flag needed to try mounting something (`/mount demo.greet`
or `/mount meta.write_a_skill` right away). Mounted directly from there,
never copied — see `skills/registry.py`'s `SkillStore.load_all` for the
load order that makes a same-named skill under a session's own
`--skills-dir` always shadow a bundled one. `uv run ftw tap` streams live
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
  Markdown/YAML/JSONL under `$FTW_HOME` — except FTW's own bundled skill
  catalog (`examples/skills`), which ships with the installation itself,
  is read-only, and is never written to or copied from. Indexes must be
  rebuildable from them.
- **Skills are task-scoped and ≤ 1,500 tokens** (body), measured with the
  project's canonical token counter, not a model-specific one.

## Environment

- Python 3.14 (system). `pynng` 0.9.0 ships a cp314 manylinux wheel.
- **uv** manages the project (no pip/Poetry): `uv sync`, `uv run pytest`,
  `uv run ftw`. Add deps with `uv add`.
- Deps: `pynng`, `pydantic`, `pyyaml`, `httpx`; dev: `pytest`, `pytest-timeout`,
  `pytest-asyncio` (`asyncio_mode = "auto"` in `pyproject.toml` — every
  `async def test_...` just works, no `@pytest.mark.asyncio` needed), `ruff`,
  `mypy`, `types-PyYAML`.
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

**The commit gate** (`scripts/check.sh`: `ruff check`, then `mypy`, then
`pytest -q` — cheapest check first, ~19s total on this codebase) runs on
every commit once enabled. Since `.git/hooks/` itself isn't
version-controlled, enabling it is a one-time, per-clone step:
`git config core.hooksPath .githooks`. `ruff check`/`mypy` run with no
project-specific rule config — both tools' own built-in defaults already
give a clean, low-noise signal on this codebase; a handful of intentional
exceptions (a best-effort `except Exception` that's documented right next
to it, a `subprocess.Popen` a test needs because that's the real type the
function under test takes, ...) are suppressed inline with a one-line
`# noqa: CODE - why` rather than turned off project-wide. `mypy` only
covers `src/ftw` (`[tool.mypy]` in `pyproject.toml`) — tests lean on
pytest fixtures and monkeypatching in ways that don't type-check cleanly
for little real benefit; `pynng` has no stubs/py.typed marker so it's
exempted via `[[tool.mypy.overrides]]`, not left as a standing error.
Deliberately excludes `tests/fuzz` (below) — that's about finding new bugs
over many runs, not regression-guarding a specific change, so it stays a
manual/opt-in step, not a per-commit one. Run `scripts/check.sh` by hand
any time without committing.

**Fuzz/property tests** (`tests/fuzz/`) target the three places arbitrary,
possibly-adversarial input reaches a parser before anything gets close to
a model: `protocol.parse_envelope` (the real wire boundary —
`bus.Replier._worker_loop` specifically catches `pydantic.ValidationError`
there, so anything else it might raise would take a worker down),
`skills/manifest.parse_skill_md` (a `SKILL.md` a user wrote or copied),
and `outputs.OutputStore` (the path-traversal defense, tested as an
end-to-end invariant against the real `read`/`save`, not just the
allowlist regex in isolation). Not hedged as temporary — kept deliberately
separate so it's cheap to remove if it stops earning its keep, not because
it's expected to. **Isolated on purpose**: a separate `fuzz` dependency
group (`hypothesis`, not part of `dev`) that a plain `uv sync`/`uv run
pytest` never installs, and `tests/fuzz` is excluded from ordinary
collection via `pyproject.toml`'s `norecursedirs` — the two together mean
nobody doing normal FTW development ever needs Hypothesis installed or
even glances at this directory. Run it with
`uv sync --group fuzz && uv run pytest tests/fuzz`, or `uv run --group
fuzz pytest tests/fuzz` for a one-off. To remove the whole capability:
`git rm -r tests/fuzz`, drop the `fuzz` group and the `tests/fuzz` entry
in `norecursedirs` from `pyproject.toml`, `uv lock` to refresh the
lockfile — nothing else in the codebase references any of it.

**What it's actually found so far**, honestly: `parse_envelope` and
`parse_skill_md` are already robust (pydantic's own JSON/Python validation
handles deep nesting, huge strings, and type confusion cleanly throughout;
the frontmatter regex isn't vulnerable to the adversarial dash patterns
that would trigger catastrophic backtracking in a vulnerable one) — these
two mostly *confirm* safety already provided by well-hardened libraries,
which is still worth having as a permanent regression guard, just not
where the yield has been. `OutputStore` is where fuzzing earned its keep:
it found a real bug on the first real run, not a contrived one — an
`output_id` that passes `_check_safe`'s character allowlist (all safe
characters, just a lot of them) but exceeds a real filesystem's ~255-byte
per-component name limit raised an uncaught `OSError` from deep inside
`read()`/`save()`, which `read_output`/`grep_output` (model-facing local
tools, no interceptor gate) didn't catch — a real crash path from
ordinary model behavior, not an adversary. Fixed by capping identifier
length in `_check_safe` itself, with a regression test
(`test_outputs.py::TestPathTraversal::test_an_absurdly_long_output_id_is_rejected_cleanly_not_an_oserror`)
alongside the fuzz test that found it.

## Layout

See §7 of `ftw_plan.md`; phases are in §8. Source goes in `src/ftw/`, tests in `tests/`.

## Git

Sole contributor. Commit and push directly to `main`; no pull requests.
Ask before force-pushing or rewriting history.
