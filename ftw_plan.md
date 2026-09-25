# FTW — Framed Task Workers (For The Win)

An **AI User Interface (AUI) to your own computer**: a personal, local-first agent framework that operates on your machines, executes deterministic and prompt-driven procedures, and stores everything it learns in files you own.

FTW replaces monolithic context accumulation with an **asynchronous NNG messaging spine**, an inspectable **Context Workbench**, and a **dual-mode skill runtime** (interactive mounting in the REPL vs. isolated delegated sub-agents).

---

## 0. Why FTW Exists

### What to Keep from Hermes Agent
- **Verification by measurement**: Measures output against ground truth (exit codes, test passes, file hashes), not LLM assertion.
- **Persistence**: Drives toward task completion rather than abandoning complex execution graphs.
- **Procedural codification**: Extracts structured, reusable procedures from successful exploration.
- **Model agnosticism**: Seamless routing across local runtimes (vLLM, llama.cpp, Ollama) and commercial APIs (DeepSeek, Nous Portal, OpenAI, Anthropic).

### What to Fix
| Hermes Failure Mode | Root Cause | FTW Architecture |
|---|---|---|
| **Context Bloat / Attention Collapse** | Memory, tool outputs, and intermediate chatter accumulate indefinitely in a single growing prompt. | **Context Workbench (§3.3)**: Token-zoned L1 cache; ephemeral items are explicitly evicted. |
| **Skill Sizing Explosion** | Skills are scoped broadly by subject (e.g., `cpp-projects`) and learning only appends new text. | **Task-Scoped Skills (§3.1)**: Strict size caps (~1,500 tokens), one task per skill, and learning rewrites rather than appends. |
| **Brittle Sub-Process Hacks** | Single context window attempts to act simultaneously as coordinator, planner, tool runner, and sub-agent. | **NNG Bus & Dual Execution (§2, §3.2)**: Brokerless out-of-process isolation for delegated runs; dynamic mounting for interactive REPL work. |
| **Late Failure Traps** | Memory and safety checks happen *after* output generation or not at all. | **Pre-Commit Interceptor Pipeline (§3.4)**: Real-time validation hooks prior to tool or skill execution. |

---

## 1. Core Principles

1. **NNG-First From Day One:** No stepping-stone abstractions like Python `asyncio.Queue`. Every component communicates over Nanomsg Next Generation (`pynng` in Python, `libnng` in C++) using `inproc://` for zero-overhead deterministic testing and `ipc://` (Unix domain sockets under `$XDG_RUNTIME_DIR/ftw/`, mode `0700`) for runtime isolation.
2. **Context as an Inspectable Workbench:** The LLM context window is not a dumping ground; it is an active L1 cache partitioned into explicit, budgeted zones. The `/context` command renders this breakdown at any point.
3. **Dual-Mode Skill Execution:**
   - **Mounted Mode (Interactive REPL):** A skill's instructions and tools load directly into the active workbench as a **frame** for multi-turn collaboration, then evict cleanly upon task completion. Multiple skills may be mounted at once, and mounts may nest. The model mounts and unmounts skills itself as needed; the user can do the same manually with `/mount` and `/unmount`.
   - **Delegated Mode (Sub-Agent):** A skill executes in an isolated, private out-of-process loop, returning only a structured, compact `Result` envelope.
4. **Skills Scoped by Task, Not Subject:** `cmake.diagnose_configure`, not `cpp-toolchain`. A task has a clear boundary and objective; a subject attracts unbounded documentation.
5. **Learning Rewrites; It Never Appends:** Updating a skill produces an atomic new version within the strict token budget.
6. **Language-Neutral Wire Protocol:** Every packet is a strict JSON envelope. A Python tool or coordinator can be rewritten in C++ or Rust without altering any other node on the bus.
7. **Strict TDD & Deterministic Testing:** Zero network calls or live LLM tokens in the test suite. All framework mechanics are validated using scripted mock providers and in-memory `inproc://` transport.
8. **Decoupled Observability:** Telemetry streams over an NNG `PUB` socket; live taps and auditors subscribe on demand with zero core runtime penalty. Durable traces are written directly, never through lossy `SUB`.
9. **User-Owned File Sovereignty:** Skills, memory items, and traces are plain Markdown, YAML, and JSONL files stored in `$FTW_HOME`. Indexes are derived caches that can be rebuilt from scratch at any time.

---

## 2. Architecture Overview

```text
                           ┌───────────────────────────────────────────┐
                           │            FTW REPL (CLI Shell)           │
                           │   Context Workbench (Zones: L1 Cache)     │
                           └─────────────────────┬─────────────────────┘
                                                 │
                                                 ▼
             =========================================================================
                                NNG MESSAGE SPINE (`pynng` / `libnng`)
              Transports: `inproc://` (Tests) | `ipc://$XDG_RUNTIME_DIR/ftw/*.ipc` (Runtime)
             =========================================================================
                 │                              │                             │
                 ├──────────────────────────────┼─────────────────────────────┤
                 ▼                              ▼                             ▼
   ┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
   │    Skill Workers (REP)    │  │    Tool Workers (REP)     │  │   Observability (SUB)     │
   │  - Delegated Task Loops   │  │  - Shell / Process Exec   │  │  - Live Console Tap       │
   │  - Isolated Contexts      │  │  - Remote SSH Exec        │  │  - JSONL Trace Logger     │
   │  - Returns Compact Result │  │  - MCP Client Bridges     │  │  - Telemetry Exporter     │
   └───────────────────────────┘  └───────────────────────────┘  └───────────────────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │    Memory Service (REP)   │
   │  - BM25 File Indexer      │
   │  - Pre-Commit HRR Cache   │
   └───────────────────────────┘
```

---

## 3. Core Concepts

### 3.1 Skills

A skill is a self-contained directory within the skill store. Its name is namespaced by domain and scoped by task:

```text
skills/
  cmake/
    diagnose_configure/
      SKILL.md          # Manifest frontmatter + concise procedure body (<= 1,500 tokens)
      evals/            # Verification test cases for regression gating
        missing_compiler.yaml
      references/       # Local on-demand technical notes (retrieved, not injected)
      tool.py           # Optional deterministic helper logic
      CHANGELOG.md      # Atomic version tracking
```

#### Manifest Schema (`SKILL.md`)

```yaml
name: cmake.diagnose_configure
version: 1
description: Diagnose failing CMake configuration and isolate root causes with evidence.
kind: prompt            # prompt | tool | composite
inputs:
  repo_dir: string
  error_text: string
outputs:
  cause: string
  fix: string
  evidence: list
calls: [tool.shell, toolchain.verify_installed]
needs:
  env: [host.current, host.compilers]
model: fast             # Model tier alias (resolved via ftw.toml)
budget:
  max_steps: 15
  max_tokens: 25000
  deadline_ms: 120000
verify:
  - check: command_ok
    cmd: "cmake -S {repo_dir} -B /tmp/ftw-verify"
    expect_exit: 0
    after_fix: true
  - critic: "Does the cited build log explicitly confirm the identified missing dependency?"
```

- **Hard Size Cap:** `SKILL.md` bodies must not exceed 1,500 tokens. A skill that nears this threshold must be split or refactored.
- **Allowlist Calls:** Callee skills and tools must be explicitly listed under `calls`.
- **Environment Decoupling:** Skills declare dependencies (`needs: env`) rather than hardcoding hostnames, absolute paths, or machine configurations.

---

### 3.2 Dual-Mode Skill Execution

FTW eliminates the friction between single-shot sub-agent isolation and interactive terminal workflows by providing two execution paths for any skill:

```
                                 [ User Task ]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      [ Mode A: Delegated ]                       [ Mode B: Mounted ]
   (Batch / Sub-Agent Execution)               (Interactive REPL Focus)
   - Dispatched via NNG `REQ`                  - Loaded via `/mount <skill>`
   - Isolated out-of-process loop              - Injected into REPL Workbench
   - Zero caller context pollution             - Multi-turn collaboration
   - Returns structured `Result` envelope       - Evicted on `/unmount` -> Leaves summary
```

#### Mode A: Delegated (Sub-Agent Execution)
Used by the coordinator or another skill to offload a bounded task:
1. The caller sends a `CALL` envelope over NNG to the target skill's IPC endpoint.
2. The skill runner initializes a **fresh, blank context** containing only: its `SKILL.md` body, inputs, caller brief ($\le 300$ tokens), and retrieved `needs` memory.
3. The callee loops until completion, verifies its outcome, and responds with a compact `Result`:
   ```json
   {
     "status": "ok",
     "summary": "Identified missing OpenSSL 3.0 headers; updated CMakeLists.txt search path.",
     "outputs": { "patch_applied": true },
     "evidence": ["grep -i openssl /tmp/ftw-verify/CMakeError.log"],
     "cost": { "tokens": 3420, "wall_time_ms": 4200 }
   }
   ```
4. The caller's context window grows **only by the size of the `Result` envelope**, never absorbing the callee's intermediate tool outputs, errors, or internal deliberation.

#### Mode B: Mounted (Interactive REPL Focus)
Used when a user wants to actively debug, iterate, or work alongside a skill:
1. A skill is mounted, either by the model (`mount_skill` tool) or by the user (`/mount cmake.diagnose_configure`).
2. The skill's instructions, schemas, and specialized tools are injected into the **Mounted Skill Zone** of the REPL's Context Workbench as a new **frame**.
3. Over the next several turns, the user and agent interactively run shell tools, analyze errors, and discuss solutions. Every turn, tool output, and pin is tagged with the frame it belongs to.
4. When work concludes, the frame is unmounted, either by the model (`unmount_skill` tool) or by the user (`/unmount [skill]`):
   - The framework prompts the model to summarize the frame into a distilled milestone (≤ 3 lines). If that call fails or is over budget, a deterministic summary is built instead (skill, commands run, exit codes, files touched).
   - The skill body and every turn, output, and pin tagged to the frame (and to its child frames) are **evicted from the active prompt**.
   - The milestone goes into the **Milestones Zone** and the session trace.

#### Frames: Multiple and Nested Mounts
Mounted skills form a **frame tree** rooted at the base REPL session:

```text
session (root)
├── cmake.diagnose_configure        # mounted by user
│   └── toolchain.verify_installed  # mounted by the model while diagnosing
└── git.bisect                      # sibling mount, active at the same time
```

- **Multiple active:** sibling frames can be mounted at the same time. All mounted skill bodies are present in the Mounted Skill Zone.
- **Nesting:** a skill mounted while another frame has focus becomes that frame's child.
- **Focus:** exactly one frame has focus: the most recently mounted, or the one selected with `/focus <skill>`. New turns are tagged to the focused frame.
- **Eviction is by subtree:** unmounting a frame evicts it and all of its descendants. Descendants are summarized first, and their milestones are folded into the parent's.
- **Invariant (tested):** `workbench_after_unmount == workbench_before_mount + milestone`. Mount/unmount must never leak frame content into the base context.
- **Budget:** the Mounted Skill Zone has a total cap (default 3,000 tokens across all frames). A mount that would exceed it is refused with a corrective message listing candidates to unmount. The model is never silently evicted from under.
- **Pinned mounts:** `/mount --pin <skill>` makes a frame user-owned. The model may not unmount it.

#### Self-Mounting vs. Manual Mounting
The model manages its own frames through tools: `find_skill(query)` (BM25 over skill names and descriptions, so the whole catalog never has to be in context), `mount_skill(name)`, `unmount_skill(name)`. These go through the Pre-Commit Interceptor like any other action (budget, allowlist).

Manual `/mount`, `/unmount`, `/focus`, and `/frames` exist because:
- **Testing and reproducibility:** deterministic scripted sessions without depending on model choices.
- **Steering:** the user corrects a wrong skill choice or forces a skill the model didn't find.
- **Protection:** pinned mounts that the model can't drop mid-task.
- **Debugging skills:** mount a newly written skill in isolation and exercise it by hand.

---

### 3.3 The Context Workbench

Rather than allowing conversation history to accumulate until truncation mechanisms trigger, the REPL context is governed like an **L1 Cache with explicit token zones**:

| Zone | Target Budget | Lifecycle | Purpose |
| :--- | :--- | :--- | :--- |
| **System Anchor** | ~300 tokens | Permanent | Core identity, protocol formatting, safety boundaries. |
| **User Memory** | ~500 tokens | Permanent | Standing user preferences, active host constraints. |
| **Mounted Skill** | 0 to 3,000 tokens (≤ 1,500 per skill) | Ephemeral (per frame) | Active skill instructions and tool declarations for every mounted frame (0 when none mounted). |
| **Milestones** | ~150 tokens | Rolling | Distilled summaries of the last N unmounted frames. |
| **Turn Horizon** | 2,000–4,000 tokens | Rolling FIFO | The last 2–4 conversational turns, each tagged with its frame. `ContextWorkbench` takes an `on_turn_evicted` callback for flushing aged-out turns to episodic traces; nothing wires it up yet (aspirational — future work). |
| **Scratchpad / Pinboard** | ~300 tokens | Dynamic | Mutable working state (e.g., target repo path, current error, active target), written by the model via `pin(key, value)` / `unpin(key)`. Pins are frame-tagged. |

**Zone order is the prompt order**, arranged for prefix/KV-cache reuse (llama.cpp, vLLM, and API prompt caching): least-volatile first, most-volatile last. Mounting or unmounting only invalidates the cache from the Mounted Skill Zone onward.

#### Tool Output Handles
Raw tool output never enters the prompt wholesale. Each result is stored out of band (`$FTW_HOME/outputs/<trace_id>/<id>`), and the Turn Horizon gets a bounded excerpt (head/tail plus exit code) and a handle. The model reads more with `read_output(id, start, end)` or `grep_output(id, pattern)`. Otherwise one build log undoes everything unmounting frees.

#### Token Counting
- **Budgets** (every workbench zone, not just the skill-size lint) always use one fixed, canonical estimate (`ftw.tokens.count_tokens`) — never a model-specific counter, so a skill's validity or whether a mount fits its budget never depends on which provider happens to be active. A per-provider counter, layered on top as a display refinement without becoming the source of truth for an enforced budget, is aspirational — not implemented today.
- **The skill size lint** (1,500 tokens) uses that same fixed, canonical counter, for the same reason.
- A skill's tool declarations count toward its Mounted Skill budget.

#### Visual Context Audit (`/context`)
Running `/context` in the REPL renders an immediate visual breakdown:

```text
FTW Context Workbench [Total: 2,510 / 8,000 max tokens]
├── System Anchor:       280 tokens  [====================] (Static)
├── User Memory:         420 tokens  [====================] (Static)
├── Mounted Skill:     1,150 tokens  [2 frames]
│   ├── cmake.diagnose_configure v1     820  (user, pinned, focus)
│   └── toolchain.verify_installed v3   330  (model, child)
├── Milestones:           60 tokens  [1 milestone]
├── Turn Horizon:        420 tokens  [2 turns active]
└── Scratchpad:          180 tokens  [repo=/home/user/src/core]
```

---

### 3.4 Verification & Pre-Commit Interceptors

Every action proposal is evaluated before execution; every action result is verified after execution:

```
               [ LLM Generates Action Proposal ]
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   Pre-Commit Interceptor Pipeline    │
            │   - Policy (jailbreaks, paths)       │
            │   - Budget Limits (tokens/steps)     │
            │   - Anti-Pattern / Memory Resonance  │
            └──────────────────┬───────────────────┘
                               │
                ALLOW / MODIFY │ BLOCK / ASK
                               │
            ┌──────────────────┴───────────────────┐
            ▼                                      ▼
    [ Execute on NNG Bus ]                 [ Halt & Prompt User / LLM ]
            │
            ▼
    [ Deterministic Verification ]
    - Command exit codes, file existence, regex proofs
```

- **Pre-Commit Interceptor Chain:** Sits between the model emitting a tool payload and the framework firing the NNG `REQ` socket. If an action violates path sandboxing or matches a known failure pattern in memory, execution is aborted and fed back as a system corrective.
- **Deterministic Verifiers:** Skills enforce success criteria via zero-LLM checks (e.g., exit code 0, regex match, file creation) before returning an `ok` status.
- **Critic Verification:** When deterministic checks are insufficient, a fresh, isolated critic model evaluates the evidence against the stated objective.

---

### 3.5 Memory Architecture

Memory is an on-demand retrieval service, never a static block injected wholesale into prompts:

| Tier | Contents | Injection Mechanism | Storage Format |
| :--- | :--- | :--- | :--- |
| **Working** | Current skill scratchpad | Active in context | In-memory process state |
| **User** | Core preferences, style constraints | Always loaded (capped at ~500 tokens) | `memory/user/*.md` |
| **Environment** | Machine hostnames, tool paths, installed packages | Loaded only via `needs: env` | `memory/env/*.md` |
| **Project** | Per-repository conventions, build setups | Retrieved on demand | `memory/project/<name>/*.md` |
| **Episodic** | Historical traces, action-result summaries | Retrieved via search | `traces/*.jsonl` |

- **Storage Engine:** Source of truth is local Markdown files with YAML frontmatter.
- **Retrieval Engine (MVP):** Keyword/BM25 indexing over memory files (fast, offline, zero-dependency).
- **Retrieval Engine (Advanced / Phase 5):** Fixed-size Holographic Reduced Representations (HRR) for sub-millisecond pre-commit resonance checking.

---

### 3.6 Practical REPL Learning

Rather than relying on complex, autonomous multi-turn background committees, FTW makes self-improvement an explicit, collaborative workflow in the REPL:

* `/remember "<fact>"`: Extracts an environment or preference rule and saves it to the appropriate file in `memory/env/` or `memory/user/`.
* `/learn`: Prompts the framework to analyze the recent Scratchpad and Turn Horizon, extract the successful procedure, format it into a valid task-scoped `SKILL.md` (enforcing the $\le 1,500$ token limit), and write it to the local skill store.
* **Atomic Versioning:** Any edit or re-learning of an existing skill triggers an atomic rewrite, bumps the skill `version`, and records a local Git commit in the skill repository.

---

## 4. NNG Wire Protocol

All bus communication utilizes Nanomsg Next Generation Scalability Protocols over standard transport URLs:
* Unit Tests: `inproc://<service_name>`
* Local Runtime: `ipc://$XDG_RUNTIME_DIR/ftw/<service_name>.ipc` (directory created `0700`; falls back to `$FTW_HOME/run/` if `XDG_RUNTIME_DIR` is unset). Never `/tmp`: a shell worker on a world-reachable socket is a local privilege hole. This also keeps paths under the ~107-byte `sun_path` limit.
* Remote / Distributed (Optional): `tcp://<host>:<port>`

### Envelope Schema (Version 1)

```json
{
  "schema_version": 1,
  "msg_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "trace_id": "4a2c0c7d-9b0e-4363-8a39-c187a4de4bb5",
  "span_id": "f8a12e34-5678-4321-abcd-ef0123456789",
  "parent_span_id": null,
  "source": "repl.master",
  "target": "worker.tool.shell",
  "type": "CALL",
  "deadline_ms": 30000,
  "idempotency_key": "c73a1112-9831-419b-a78b-9d45e0d37e21",
  "brief": "Run CMake configure to reproduce missing dependency error",
  "payload": {
    "action": "run_command",
    "args": {
      "argv": ["cmake", "-S", ".", "-B", "build"]
    }
  }
}
```

`payload` is a discriminated union keyed on `type`; Pydantic models are the source, and JSON Schemas are exported to `schemas/` for non-Python workers.

#### Protocol Message Types
- `CALL`: Request execution of a tool or skill.
- `RESULT`: Structured output and execution summary.
- `ERROR`: Standardized execution failure envelope.
- `ASK`: Callee needs an answer (human confirmation, missing input) before it can continue. Carries a `resume_token`.
- `ANSWER`: Caller's response to an `ASK`, sent as a new request carrying the `resume_token`.
- `CANCEL`: Abort an in-flight call by `span_id` (sent on the control channel, see below).
- `PROGRESS`: Optional intermediate status for long calls (published as an event, never required for correctness).
- `EVENT`: Telemetry records broadcast over NNG `PUB`.
- `INTERCEPT`: Pre-commit block or modification event.

### Socket Patterns & Semantics

| Channel | Pattern | Notes |
| :--- | :--- | :--- |
| Calls to tools/skills | `REQ`/`REP` using **pynng contexts** | Contexts let one socket carry many concurrent requests; a worker serves several calls at once. |
| Control (cancel, health) | `REQ`/`REP` on a separate `<service>.ctl.ipc` | A busy call channel can't block a `CANCEL`. |
| Live telemetry | `PUB`/`SUB` on `events.ipc` | Lossy by design. For taps only. |
| Durable traces | Written directly by the process that owns the trace (or `PUSH`/`PULL` to a logger) | Never through `SUB`. Traces feed milestones and episodic memory, so they can't drop events. |

- **REQ resend is a correctness hazard.** NNG `REQ` resends an unanswered request after `resend_time` (default 60 s). A 120 s skill call would execute twice. The bus wrapper sets `resend_time` from the call's `deadline_ms` (or disables it), and workers **dedupe on `idempotency_key`**: a repeated key returns the cached result or joins the in-flight call. Both are covered by tests in Phase 0.
- **Mid-call interaction:** a strict REQ/REP exchange can't carry a question back to the caller. A worker that needs input replies with `ASK` (its loop suspended under `resume_token`); the caller answers with a fresh `ANSWER` request. Many `ASK`/`ANSWER` rounds may occur before the final `RESULT`/`ERROR`.
- **Topic framing on PUB:** SUB filtering is byte-prefix based and every JSON body starts with `{`, so event messages are framed as `<topic>\0<json>` (e.g. `event.tool.shell.exit\0{...}`). This is part of schema version 1.
- **Deadlines:** every receive has a timeout derived from `deadline_ms`. A timeout surfaces as an `ERROR` envelope with `code: "deadline_exceeded"`, never a hang.

### Service Discovery & Supervision
- **Addressing:** a `target` such as `worker.tool.shell` maps to an endpoint by convention (`<runtime_dir>/worker.tool.shell.ipc`), with overrides in `ftw.toml`. Tools and skills share one registry of addresses.
- **Supervision:** the REPL launches the configured workers as child processes at startup, health-checks them over the control channel, restarts on crash, and removes their socket files on shutdown. Stale socket files from a crashed run are detected (connect fails) and removed at startup. A standalone `ftw up` may come later. **Not yet built as described** — see §9's "Config-driven worker supervision."

---

## 5. Observability & Tracing

- **NNG `PUB`/`SUB` Tap:** All lifecycle transitions, model requests, interceptor decisions, frame mount/unmount, and tool executions publish structured `EVENT` envelopes to `<runtime_dir>/events.ipc`.
- **Zero Overhead When Unobserved:** If no tap is listening, NNG discards messages immediately without blocking execution.
- **Local Tap CLI (`ftw tap`):** Streams live, colorized, tree-indented execution graphs to an auxiliary terminal.
- **Durable Logging:** The owning process appends events to `$FTW_HOME/traces/<date>/<trace_id>.jsonl` directly (see §4: not via lossy `SUB`).

---

## 6. Model Providers

All model access goes through `IModelProvider`. It is in-process (not a bus worker) for now, and publishes `EVENT`s for every request. One normalized internal representation for messages and tool calls; each provider adapter translates to and from its wire format (native function calling where supported, JSON-in-text fallback otherwise).

| Provider | Adapter | Base URL | Credential |
| :--- | :--- | :--- | :--- |
| Mock (tests) | `MockModelProvider` (scripted turns and tool calls) | — | — |
| DeepSeek | `OpenAICompatibleProvider` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Nous Research Portal | `OpenAICompatibleProvider` | `https://inference-api.nousresearch.com/v1` | `NOUS_API_KEY` |
| Local (llama.cpp / vLLM / Ollama) | `OpenAICompatibleProvider` | e.g. `http://localhost:8080/v1` | none |
| Anthropic (later) | native adapter | — | `ANTHROPIC_API_KEY` |

Keys are read from the environment only, never written to `ftw.toml`, traces, or events. Model tiers are aliases resolved in `ftw.toml`:

```toml
[providers.deepseek]
kind = "openai_compatible"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[providers.nous]
kind = "openai_compatible"
base_url = "https://inference-api.nousresearch.com/v1"
api_key_env = "NOUS_API_KEY"

[tiers]
fast  = { provider = "deepseek", model = "<model-id>" }
smart = { provider = "nous",     model = "<model-id>" }
```

Model IDs are filled in at setup time from each provider's `/models` listing. Live-provider checks live in a separate, opt-in smoke-test command, never in the test suite.

---

## 7. Project Layout

```text
ftw/
├── pyproject.toml               # uv-managed (deps: pynng, pydantic, pyyaml, httpx; dev: pytest)
├── uv.lock
├── ftw.toml                     # Local runtime configuration, providers, and model tiers
├── schemas/                     # Exported JSON schemas for polyglot workers
├── src/ftw/
│   ├── __init__.py
│   ├── protocol.py              # Envelope, Result, Error, Ask/Answer/Cancel models
│   ├── bus.py                   # NNG socket wrappers (Req/Rep w/ contexts, Pub/Sub), dedupe, cleanup
│   ├── runtime.py               # Runtime dir, service registry, worker supervision
│   ├── workbench.py             # Context Workbench (zones, token accounting, rendering)
│   ├── frames.py                # Frame tree: mount/unmount, focus, subtree eviction, milestones
│   ├── agent_loop.py            # Model → interceptors → bus → result loop (shared by REPL and delegated runner)
│   ├── outputs.py               # Out-of-band tool output store and handles
│   ├── tokens.py                # Canonical counter + per-provider counters
│   ├── providers/               # IModelProvider, MockModelProvider, OpenAICompatibleProvider
│   ├── repl/                    # REPL loop, readline integration, slash commands
│   ├── skills/                  # Manifest loader, size linter, registry, find_skill, delegated runner
│   ├── tools/                   # Isolated tool workers (ShellWorker, SSHWorker)
│   ├── intercept/               # Pre-commit hook pipeline (policy, budget, anti-pattern checks)
│   ├── verify/                  # Deterministic checks and critic verifiers
│   └── memory/                  # File-backed storage, BM25 indexing, tier managers
└── tests/
    ├── conftest.py              # Inproc NNG fixtures and mock provider setup
    ├── test_protocol.py         # Envelope serialization and schema adherence
    ├── test_bus.py              # Inproc transport, timeouts, resend/dedupe, ASK/ANSWER, cancel
    ├── test_workbench.py        # Token zoning, ordering, truncation
    ├── test_frames.py           # Nested/multiple mounts, focus, eviction invariant
    ├── test_skills.py           # Manifest parsing, size linting, find_skill, delegation
    ├── test_workers.py          # Shell and tool execution over NNG sockets
    └── test_repl_flow.py        # Multi-turn interaction and offline REPL simulation
```

Tooling: `uv sync`, `uv run pytest`, `uv run ftw`.

---

## 8. Phased Implementation Roadmap

Each phase is executed with strict Test-Driven Development (TDD) and finishes with a working, runnable milestone.

### Phase 0: Foundations, Protocol & NNG Transport
- `uv` project skeleton, `pytest` wired up.
- Implement all envelope types (`CALL`, `RESULT`, `ERROR`, `ASK`, `ANSWER`, `CANCEL`, `PROGRESS`, `EVENT`, `INTERCEPT`) in `protocol.py` with Pydantic; export JSON Schemas.
- Implement socket management in `bus.py` wrapping `pynng` (`Req0`/`Rep0` with contexts, `Pub0`/`Sub0` with topic framing), deadline-driven timeouts, `resend_time` control, and `idempotency_key` dedupe.
- Runtime directory handling (`$XDG_RUNTIME_DIR/ftw/`, `0700`) and stale-socket cleanup.
- **Deliverable:** Contract test suite over `inproc://`, plus a small `ipc://` integration test, covering timeouts, duplicate-suppression, and an `ASK`/`ANSWER` round trip.

### Phase 1: Providers, Agent Loop, Workbench & Shell Worker
- `IModelProvider`, scriptable `MockModelProvider`, and `OpenAICompatibleProvider` (DeepSeek and Nous Portal configured in `ftw.toml`).
- `ContextWorkbench` with token accounting and zone ordering (`System`, `User`, `MountedSkill`, `Milestones`, `TurnHorizon`, `Scratchpad`); `pin`/`unpin` tools.
- `agent_loop.py`: the model → interceptor → bus → result cycle.
- `ShellToolWorker` as an independent NNG service, with out-of-band output handles (`read_output`, `grep_output`).
- Minimal interceptor pipeline with one rule: **confirm every shell command with the user** (until Phase 4 policy exists).
- REPL with `/context`, `/clear`, `/exit`; worker supervision; `ftw tap`; direct trace writing.
- **Deliverable:** Talk to DeepSeek or Nous in the REPL, have it run confirmed shell commands over NNG, and watch `/context` and `ftw tap` in real time.

### Phase 2: Skills & Mounted Frames (headline feature)
- `SKILL.md` parser, frontmatter validator, and canonical-counter 1,500-token linter.
- `frames.py`: frame tree with multiple and nested mounts, focus, subtree eviction, pinned mounts, Mounted Skill Zone total budget.
- Milestone distillation on unmount with a deterministic fallback; Milestones Zone.
- Model tools: `find_skill`, `mount_skill`, `unmount_skill`. User commands: `/mount [--pin]`, `/unmount`, `/focus`, `/frames`.
- **Invariant tests:** `after_unmount == before_mount + milestone` for single, sibling, and nested frames, and for mixed model/user mounts.
- **Deliverable:** The model finds and mounts a skill (and a nested helper), works multi-turn with shell tools, and unmounts. `/context` shows the footprint back at baseline plus milestones.

### Phase 3: Delegated Skill Runner
- Isolated out-of-process skill worker built on the same frame/agent-loop code with a fresh workbench.
- Compact `Result` envelopes; `ASK` passthrough to the REPL; `CANCEL` from Ctrl-C.
- **Deliverable:** A mounted skill or the REPL delegates a sub-task; caller context grows only by the `Result`.

### Phase 4: Interceptors & Deterministic Verifiers
- `PreCommitInterceptor` pipeline (path sandboxing, dangerous command blocklist, step budget caps, `calls` allowlist), replacing blanket confirmation with policy-based `ALLOW`/`ASK`/`BLOCK`.
- Deterministic `Verifier` pipeline (command exit codes, file existence, regex matches).
- **Deliverable:** Shell worker halts on dangerous commands; failed actions retry deterministically up to budget limit.

### Phase 5: Practical Memory & REPL Learning
- File-backed memory storage (`memory/user/`, `memory/env/`, `memory/project/`).
- BM25 retrieval over memory markdown files (generalizing the Phase 2 `find_skill` index).
- `/remember` and `/learn` REPL commands to synthesize procedures from recent turns into new task-scoped `SKILL.md` files.
- Track skill revisions automatically via local Git commits.
- **Deliverable:** Agent codifies a multi-step shell procedure into a reusable, verified skill directly from the REPL session.

### Phase 6: Polyglot Expansion & Holographic Memory
- Reference C++17 worker implementation using `libnng` conforming to `schemas/`.
- Experimental Holographic Reduced Representation (HRR) vector index for sub-millisecond pre-commit anti-pattern matching.
- **Deliverable:** Polyglot bus demonstration: Python REPL coordinates task execution across both Python and C++ workers.

---

## 9. Future Enhancements (Not Yet Scheduled)

Candidate work that doesn't fit a specific phase above — cross-cutting or dependent on how the framework feels in practice once there's more to mount and run.

- **Streaming model output.** `IModelProvider.complete()` is a single blocking call today: on a large reasoning model or a loaded local runtime, the REPL sits frozen for several seconds with no feedback. The `PROGRESS` envelope type (§4) is already reserved for this. Doing it properly touches three places at once — a streaming variant of the provider interface, `MockModelProvider` scripting token deltas deterministically (not just final responses) so it stays testable, and the REPL rendering partial output as it arrives — so it's sized as its own piece of work, not a small patch alongside something else.
- ~~A real Ctrl-C trigger for CANCEL~~ — **done, including the idle prompt.** The bus (`Requester`/`Replier`/`DispatchRouter`), `AgentLoop`, `skills/runner.py`, and the REPL all moved to `asyncio`; `Requester.call()` uses `asend()`/`arecv()` (NNG's real `nng_aio`/`nng_aio_cancel` API, not the old blocking `send()`/`recv()` that swallowed a pending SIGINT), and `repl/session.py` cancels each turn's own `asyncio.Task` on a real SIGINT. A real Ctrl-C at an idle prompt (no turn in flight) and while answering a mid-turn confirmation prompt both work too now — `repl/session.py`'s `_blocking_read` reads on the main thread with Python's plain default SIGINT handler temporarily active, rather than bridging through `asyncio.to_thread` (which turned out to actively deadlock `asyncio.run()`'s own shutdown, not just fail to cancel cleanly). `tests/test_bus.py::TestRecvIsInterruptible` is no longer `xfail`. See `CLAUDE.md`'s status section for the full mechanism and how it was verified against the real binary, not just unit tests.

### From an external review (post-Phase-3), not yet acted on

Surfaced by a second model reviewing the codebase and verified against the actual code before being recorded here (not taken on faith). The gate item from the same review is already done — see `CLAUDE.md`'s Testing section — and two small hardening items (`Replier` liveness-checked rebind, a bounded idempotency cache) are already fixed. These are the ones still open:

- **Delegated leaf-vs-recursive (open decision, deferred on purpose).** `skills/runner.py`'s `SkillRunnerWorker._handle_call` constructs its `AgentLoop` with no `frame_tree` and no `skill_runner_target` (`runner.py:160-170`), so a delegated run's tool list is `run_command`/`pin`/`unpin`/`read_output`/`grep_output`/`submit_result` only — no `find_skill`/`mount_skill`/`delegate_skill`. §3.2 Mode A above describes delegation without saying this, so a reader would assume it's recursive. It isn't, today, and that's deliberately undecided rather than wrong: making it recursive is real feature work (recursion-depth limits, budget propagation across levels, nested `DelegatedSuspension`/`ASK` relay), not a small patch. Revisit when Phase 4/5 needs an answer, not before.
- **Config-driven worker supervision.** §4's Service Discovery & Supervision above describes the target end state; today's reality is `repl/cli.py` + `agent_loop.py` hardcoding worker spawn, `DispatchRouter` wiring, tool specs, and call deadlines directly, with only a one-shot ~0.5s startup-alive poll (`_wait_for_worker_or_crash`) — no `[workers]` table in `ftw.toml`, no address overrides, no runtime health-check, no restart-on-crash. Adding a worker today means editing several core files; a `[workers]` table plus a generic spawn/supervise loop would make that configuration instead.
- **Async interceptor, ahead of Phase 4.** `PreCommitInterceptor.evaluate()` (`intercept/`) is a plain synchronous `def` returning a plain outcome; `ConfirmShellCommands` is the only rule that exists. Phase 4's critic/resonance rules will want to make real (async) calls of their own — an LLM critic call, a memory-resonance lookup. Converting the interface to `async def` is cheap now, with one trivial rule to update; doing it after Phase 4 rules exist means retrofitting async through code that was written sync.
- **Skill budget fields parsed but not enforced.** `SkillManifest.budget` (`skills/manifest.py`) has `max_steps`, `max_tokens`, and `deadline_ms`; only `max_steps` is actually read back out by `AgentLoop`. A skill declaring a token or wall-clock budget today gets no enforcement of either.
- **No retention for `outputs/` or `traces/`.** Both grow without bound in a long-lived `$FTW_HOME` — nothing prunes old trace-of-the-day directories or tool-output blobs. Not urgent (files, not memory — the process doesn't leak), but worth a policy (age-based? size-capped?) before this framework is anyone's daily driver.

### Concurrent delegated sub-agents (mixture-of-agents)

Not from the review — came up discussing whether several sub-agents could run at once and have their outputs assembled into one report. Checked the actual dispatch path rather than guessing: **sequential only, today.** `AgentLoop._step_loop` walks a model response's tool calls with a plain `for` loop, `await`-ing each `_run_tool_call` fully before starting the next (`agent_loop.py:479-489`) — three `delegate_skill` calls in one turn run one after another, not in parallel. The receiving side is already capable of concurrency (`skills/runner.py`'s worker is served by a `Replier` with `num_workers=4` concurrent `asyncio.Task`s, so it can already field several `delegate_skill` `CALL`s arriving at once); the gap is entirely on the caller side, which holds one `Requester` at a time (`bus.py`'s own docstring: "one call in flight at a time; open several instances for concurrency").

Making it work: dispatch a step's independent tool calls via `asyncio.gather` instead of the `for` loop (each result already carries its own `tool_call_id`, so reassembly doesn't need call-order tracking), backed by either multiple `Requester`s or a dispatch layer that can hold several calls open to one target. No new assembly machinery needed — N results land back as tool messages in the same turn, and the model's next response is naturally the synthesized report, same as after any single delegated call today. Smaller than the leaf-vs-recursive question above (doesn't change what a delegated run *can do*, just how many run at once) — but still real work, not a quick patch, so deferred alongside it.
