#!/usr/bin/env bash
# The in-repo commit gate (CLAUDE.md's Testing section): lint, then types,
# then the full test suite - in that order, so the cheapest check fails
# fastest. Wired in as a pre-commit hook via .githooks/pre-commit (one-time
# setup: `git config core.hooksPath .githooks`); also runnable by hand.
# Deliberately excludes tests/fuzz - see CLAUDE.md for why that stays opt-in.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "--- ruff check ---"
uv run ruff check .

echo "--- mypy ---"
uv run mypy

echo "--- pytest ---"
uv run pytest -q
