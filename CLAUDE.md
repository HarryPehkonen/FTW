# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

FTW (Framed Task Workers) is a local-first agent framework: a REPL whose LLM
context is an explicitly budgeted "workbench", skills that can be **mounted**
into that context and **unmounted** to free the space, and workers that talk
over an NNG message bus. The design lives in `ftw_plan.md`; read it before
making architectural changes.

**Status:** Phase 0 (protocol + bus + runtime dir) and Phase 1 (providers,
workbench, agent loop, shell worker, interceptor, REPL) are done. Phase 2
(skills and mounted frames — the headline feature) is next. See §8 of
`ftw_plan.md` for the phase list.

Run it: `uv run ftw` (needs a real key for the `fast`/`smart` tiers in
`ftw.toml` — `DEEPSEEK_API_KEY` / `NOUS_API_KEY` — until Phase 2's skills
give it something more interesting to do than chat). `uv run ftw tap`
streams live events from a running session in another terminal.

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
