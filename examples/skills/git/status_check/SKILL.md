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
3. Summarize: how many files changed, whether there are untracked files,
   and whether the tree is clean. Don't make any changes — this skill only
   reads state.
