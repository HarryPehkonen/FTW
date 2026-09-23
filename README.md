# FTW — Framed Task Workers

A personal, local-first agent framework. FTW talks to a model over an
async **NNG message bus**, keeps its context in an inspectable, budgeted
**Context Workbench** instead of one ever-growing prompt, and lets you
**mount and unmount skills** into that context on demand — multiple at
once, nested, freed the moment you're done with them.

See [`ftw_plan.md`](ftw_plan.md) for the full design, and
[`CLAUDE.md`](CLAUDE.md) for the project's working conventions.

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
uv run ftw                                    # start the REPL
uv run ftw --skills-dir examples/skills       # ...with two ready-to-mount example skills
uv run ftw tap                                # in another terminal: stream live events from a running session
```

## The REPL

| Command | Does |
| :--- | :--- |
| `/context` | Shows the current token budget, zone by zone. |
| `/clear` | Resets the turn history. |
| `/mount [--pin] <skill>` | Mounts a skill; `--pin` stops the model from unmounting it. |
| `/unmount [skill]` | Unmounts a skill (or, with no name, whichever is focused), distilling its work into a milestone. |
| `/focus [skill]` | Switches which mounted skill new turns and mounts attach to; no argument returns focus to the root session. |
| `/frames` | Shows the mounted-skill tree. |
| `/exit` | Quits. |

The model can also find, mount, and unmount skills itself (`find_skill`,
`mount_skill`, `unmount_skill`), and manages its own scratchpad
(`pin`/`unpin`) and tool-output handles (`read_output`/`grep_output`) —
all without your involvement unless you step in with the commands above.
Shell commands the model proposes are confirmed with you before they run.

The prompt supports normal line editing and history (arrow keys, `Ctrl-R`
search) on POSIX systems.

## Delegating a sub-task

The model can hand a bounded task to a skill with `delegate_skill(name,
brief, inputs)`. That skill runs to completion in its own process with its
own fresh context — none of its intermediate steps ever reach the current
conversation, only a short structured result (a summary, plus any
outputs/evidence it reports) comes back. This is the tool to reach for when
a task is well-scoped enough to run unsupervised rather than worked through
turn by turn.

If the delegated task needs to run a shell command, it asks for
confirmation exactly like an ordinary command would — that question is
relayed back to you here, and your answer resumes the delegated run right
where it paused.

## Skills

A skill is a directory containing a `SKILL.md`: YAML frontmatter (name,
description, and other metadata) followed by a Markdown body — the
procedure that gets injected into context once mounted. Bodies are capped
at 1,500 tokens; a skill that outgrows that needs to be split.

```yaml
---
name: cmake.diagnose_configure
description: Diagnose failing CMake configuration and isolate root causes with evidence.
kind: prompt
---

Reproduce the failure, grep the error log, report cause and fix.
```

Point `--skills-dir` at a directory of these (default `<ftw-home>/skills`)
and the model can find and mount them by name or by searching. Two
examples live in [`examples/skills`](examples/skills).

## Development

```bash
uv run pytest                        # full suite, a few seconds
uv run scripts/export_schemas.py     # regenerate schemas/*.schema.json after touching protocol.py
```

TDD throughout: no network calls or live model tokens anywhere in the test
suite.

## Project layout

```text
src/ftw/
├── protocol.py            envelope schema (CALL, RESULT, ERROR, ASK, ANSWER, CANCEL, PROGRESS, EVENT, INTERCEPT)
├── bus.py                 NNG socket wrappers: Req/Rep over contexts, Pub/Sub, dedupe, cleanup
├── runtime.py             runtime directory + inproc/ipc address resolution
├── tokens.py              the canonical, provider-independent token counter
├── workbench.py           the Context Workbench: budgeted zones, prompt rendering
├── frames.py              the mounted-skill frame tree: nesting, focus, subtree eviction, milestones
├── skills/                SKILL.md parsing, the size lint, the skill store (find_skill via BM25), and the delegated-run worker (runner.py)
├── outputs.py             out-of-band tool-output store (read_output/grep_output handles)
├── agent_loop.py          model -> interceptor -> bus -> result cycle; also drives delegated (sub-task) completions
├── intercept/             Pre-Commit Interceptor pipeline
├── providers/             IModelProvider, MockModelProvider, OpenAICompatibleProvider
├── tools/shell.py         the shell tool worker, as an independent NNG service
├── trace.py               durable JSONL trace writing
├── observability/tap.py   ftw tap: the live event viewer
├── config.py              ftw.toml loading and provider/tier resolution
└── repl/                  the REPL loop (session.py) and its CLI wiring (cli.py)

tests/             one test file per module above
examples/skills/   a couple of skills ready to mount
schemas/           generated JSON Schemas for polyglot (non-Python) workers
```

## License

See [`LICENSE`](LICENSE).
