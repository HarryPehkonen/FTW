# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

FTW (Framed Task Workers) is a local-first agent framework: a REPL whose LLM
context is an explicitly budgeted "workbench", skills that can be **mounted**
into that context and **unmounted** to free the space, and workers that talk
over an NNG message bus. The design lives in `ftw_plan.md`; read it before
making architectural changes.

**Status:** planning. No code yet. Don't write implementation code until the
user says the planning stage is over.

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
- Planned deps: `pynng`, `pydantic`, `pyyaml`, `httpx`; dev: `pytest`.
- Providers: DeepSeek and Nous Research Portal, both via the OpenAI-compatible
  adapter (see §6 of the plan). Live-provider checks are opt-in smoke tests,
  never part of `pytest`.

## Layout

See §7 of `ftw_plan.md`; phases are in §8. Source goes in `src/ftw/`, tests in `tests/`.

## Git

Sole contributor. Commit and push directly to `main`; no pull requests.
Ask before force-pushing or rewriting history.
