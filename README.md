# FTW — Framed Task Workers

A personal, local-first agent framework. FTW talks to a model over an
async **NNG message bus**, keeps its own context in an inspectable,
budgeted **Context Workbench** instead of one ever-growing prompt, and lets
you **mount and unmount skills** into that context on demand — multiple at
once, nested, freed again the moment you're done with them — so a long
session never has to mean a bloated one.

See [`ftw_plan.md`](ftw_plan.md) for the full design and roadmap, and
[`CLAUDE.md`](CLAUDE.md) for the project's working conventions.

## Status

- **Phase 0** — envelope protocol, NNG bus (Req/Rep with contexts, Pub/Sub),
  runtime directory handling. Done.
- **Phase 1** — model providers, the Context Workbench, the agent loop,
  the shell tool worker, a minimal confirm-every-command interceptor, the
  REPL, live event tap, durable tracing. Done.
- **Phase 2** (next) — skills and mounted frames: `/mount`, `/unmount`,
  nested and simultaneous mounts, self-mounting by the model.

155 tests, TDD throughout, no network calls or live model tokens in the
test suite.

## Quickstart

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set a provider key and point a tier at a real model in `ftw.toml` (DeepSeek
and the Nous Research Portal are pre-wired; any OpenAI-compatible local
runtime — llama.cpp, vLLM, Ollama — works too):

```bash
export DEEPSEEK_API_KEY=...   # or NOUS_API_KEY
```

```bash
uv run ftw          # start the REPL
uv run ftw tap       # in another terminal: stream live events from a running session
```

In the REPL: `/context` shows the current token budget by zone, `/clear`
resets the turn history, `/exit` quits. Shell commands the model proposes
are confirmed with you before they run.

## Development

```bash
uv run pytest                        # full suite, ~2.5s
uv run scripts/export_schemas.py     # regenerate schemas/*.schema.json after touching protocol.py
```

## Project layout

```text
src/ftw/
├── protocol.py            envelope schema (CALL, RESULT, ERROR, ASK, ANSWER, CANCEL, PROGRESS, EVENT, INTERCEPT)
├── bus.py                 NNG socket wrappers: Req/Rep over contexts, Pub/Sub, dedupe, cleanup
├── runtime.py             runtime directory + inproc/ipc address resolution
├── tokens.py              the canonical, provider-independent token counter
├── workbench.py           the Context Workbench: budgeted zones, prompt rendering
├── outputs.py             out-of-band tool-output store (read_output/grep_output handles)
├── agent_loop.py          model -> interceptor -> bus -> result cycle
├── intercept/             Pre-Commit Interceptor pipeline
├── providers/             IModelProvider, MockModelProvider, OpenAICompatibleProvider
├── tools/shell.py         the shell tool worker, as an independent NNG service
├── trace.py               durable JSONL trace writing
├── observability/tap.py   ftw tap: the live event viewer
├── config.py              ftw.toml loading and provider/tier resolution
└── repl/                  the REPL loop (session.py) and its CLI wiring (cli.py)

tests/      one test file per module above, plus test_repl_flow.py, test_repl_cli.py
schemas/    generated JSON Schemas for polyglot (non-Python) workers
```

## License

See [`LICENSE`](LICENSE).
