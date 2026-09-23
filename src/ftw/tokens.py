"""The canonical token counter (ftw_plan.md §3.3, §3.1).

Budgets (workbench zones, the 1,500-token skill-body cap) are measured with
this counter, not a model-specific one — a skill's validity, or whether a
mount fits its budget, must not depend on which provider happens to be
active. Per-provider counters (when a provider exposes its own tokenizer)
are a display refinement layered on top later; they are never the source of
truth for an enforced budget.

The count itself is a deliberate, documented approximation: words and
punctuation each count as one token, which tracks BPE tokenizers closely
enough for budgeting without depending on any of them.
"""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Deterministic, offline token estimate for ``text``."""
    return len(_TOKEN_PATTERN.findall(text))
