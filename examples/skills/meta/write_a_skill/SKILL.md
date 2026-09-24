---
name: meta.write_a_skill
description: Explains the SKILL.md format and walks through writing a new skill, step by step.
kind: prompt
---

# Writing a Skill

A skill is a directory containing one `SKILL.md`: YAML frontmatter, then
a `---`, then a Markdown body. Only the body counts against the size
cap (see below) — the frontmatter is read by the framework, never sent
to a model verbatim.

## Frontmatter fields

- `name` (required) — dotted, e.g. `cmake.diagnose_configure`. This is
  the identifier used to mount/find the skill; the directory layout on
  disk is just for the author's own organization and doesn't have to
  match it.
- `description` (required) — one sentence. This is what `find_skill`'s
  search matches against, so use the words someone would actually
  search with, not just a category label.
- `kind` — `prompt` (the common case: pure instructions, no special
  execution mode), `tool`, or `composite`. Default `prompt`.
- `inputs` — a `{name: type}` map documenting what the caller is
  expected to supply in the brief (e.g. `repo_dir: string`). Informational
  today, not yet enforced.
- `calls` — which tools/workers this skill is expected to use (e.g.
  `[tool.shell]`). Informational, for a human or another skill reading
  the catalog — not an enforced allowlist.
- `model` — pins a specific tier/model for this skill's own work, if it
  needs something other than whatever tier is active by default.

Everything else (`budget`, `needs`, `verify`) is for more advanced
cases — skip them unless you have a specific reason to set one.

## The body

Plain Markdown: the actual procedure. Write it as instructions to
whoever mounts the skill, not as a description of what it does — "Run
X, then check Y" rather than "This skill runs X and checks Y". Keep it
concrete and step-by-step rather than abstract; a skill that just
restates its own description isn't pulling weight over the description
alone.

The body is capped at 1,500 tokens (roughly 1,100 words), measured with
FTW's own canonical counter, not any particular model's tokenizer. A
skill that's outgrowing this needs to be split — either into a parent
skill that mounts a nested child for the detailed part, or into two
separate skills the model finds independently.

## Where it goes

Save it as `<ftw-home>/skills/<any-path-you-like>/SKILL.md` — the
directory path itself is just organization; only `name` in the
frontmatter matters for lookup. `<ftw-home>` defaults to `~/.ftw`
(override with `--ftw-home`, or `--skills-dir` for just the skills
path).

A separate, read-only bundled catalog also ships with FTW itself and is
always searched too, with no configuration needed. A skill saved to
`<ftw-home>/skills` with the same `name` as a bundled one always wins —
that's how you customize a bundled skill: don't edit the bundled copy,
save your own version under the same name in your own skills directory.

## A minimal example

```yaml
---
name: git.status_check
description: Check a git repository's working tree status and summarize what's changed.
kind: prompt
inputs:
  repo_dir: string
calls: [tool.shell]
---

# Check Git Status

1. Run `git -C {repo_dir} status --short` to see changed, added, and
   untracked files.
2. Run `git -C {repo_dir} log -1 --oneline` to see the current commit.
3. Summarize: how many files changed, whether there are untracked
   files, and whether the tree is clean. Don't make any changes — this
   skill only reads state.
```

Note what makes this a good skill body: it's a short, ordered procedure
with concrete commands, not a paragraph of prose about git. The
description alone tells a search "this is for checking git status"; the
body tells the model exactly what to run and what to report once
mounted.

## A skill meant for delegation

A skill that's well-scoped enough to run unsupervised (via
`delegate_skill`, not just mounted into the current conversation) needs
one more thing: it must end by calling `submit_result` with a status and
a short summary — that's what a delegated run's caller actually receives
back, not any of the skill's own intermediate steps. Say this
explicitly in the body:

```markdown
When you've confirmed the result, call `submit_result` with `status`
("ok" or "error") and a one- or two-sentence `summary` — that's all the
caller will see.
```

Everything else about the skill (frontmatter, body style, size cap)
works exactly the same whether it's meant to be mounted directly or
delegated — `submit_result` is the only thing that changes.
