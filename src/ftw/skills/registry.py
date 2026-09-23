"""SkillStore: loads SKILL.md files from a directory and answers
``find_skill`` queries via BM25 over name + description (ftw_plan.md §3.2)
— the model never needs the whole catalog in context to pick a skill.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ftw.skills.manifest import SkillManifest, SkillParseError, lint_skill_body, parse_skill_md

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_BM25_K1 = 1.5
_BM25_B = 0.75


class SkillNotFound(Exception):
    pass


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_rank(corpus: dict[str, str], query: str) -> list[str]:
    """Ranks corpus keys by BM25 score against ``query``, best first,
    dropping anything that scores zero (no query term present at all)."""
    doc_tokens = {name: _tokenize(text) for name, text in corpus.items()}
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return []

    avg_len = sum(len(toks) for toks in doc_tokens.values()) / n_docs
    doc_freq: dict[str, int] = {}
    for toks in doc_tokens.values():
        for term in set(toks):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    query_terms = _tokenize(query)
    scores: list[tuple[str, float]] = []
    for name, toks in doc_tokens.items():
        doc_len = len(toks)
        term_freq: dict[str, int] = {}
        for t in toks:
            term_freq[t] = term_freq.get(t, 0) + 1

        score = 0.0
        for term in query_terms:
            f = term_freq.get(term, 0)
            if f == 0:
                continue
            n_t = doc_freq.get(term, 0)
            idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
            score += idf * (f * (_BM25_K1 + 1)) / (f + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avg_len))
        if score > 0:
            scores.append((name, score))

    scores.sort(key=lambda pair: pair[1], reverse=True)
    return [name for name, _ in scores]


class SkillStore:
    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._cache: dict[str, tuple[SkillManifest, str]] | None = None

    def load_all(self) -> dict[str, tuple[SkillManifest, str]]:
        """Re-scans the skill store from disk. Any malformed or oversized
        skill fails the whole load, loudly, with the offending path —
        never silently skipped."""
        result: dict[str, tuple[SkillManifest, str]] = {}
        for path in sorted(self._root.glob("**/SKILL.md")):
            try:
                manifest, body = parse_skill_md(path.read_text())
                lint_skill_body(body)
            except SkillParseError as exc:
                raise type(exc)(f"{path}: {exc}") from exc
            result[manifest.name] = (manifest, body)
        self._cache = result
        return result

    def refresh(self) -> None:
        self.load_all()

    def _ensure_loaded(self) -> dict[str, tuple[SkillManifest, str]]:
        if self._cache is None:
            self.load_all()
        assert self._cache is not None
        return self._cache

    def get(self, name: str) -> tuple[SkillManifest, str]:
        cache = self._ensure_loaded()
        if name not in cache:
            raise SkillNotFound(f"no skill named {name!r}")
        return cache[name]

    def find(self, query: str, *, top_k: int = 5) -> list[str]:
        cache = self._ensure_loaded()
        corpus = {name: f"{name.replace('.', ' ')} {manifest.description}" for name, (manifest, _) in cache.items()}
        return _bm25_rank(corpus, query)[:top_k]
