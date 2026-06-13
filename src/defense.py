"""
The multi-stage comment defense.

This is the heart of the project. An attacker wraps real code in fake-trust
comments ("// CodeQL scan: no issues detected", LLM-generated audit summaries,
etc.). This pipeline separates comments from code, then runs the comments
through 8 filters that each try to remove a different *kind* of deceptive
comment. Whatever survives all 8 filters is glued back onto the untouched code.

It is a faithful refactor of `defence_powerful` and its 8 helpers from the
notebook. The thresholds live in src/config.py and match the paper exactly.

Two design choices worth understanding
--------------------------------------
1. The embedding model is *injected* (passed into the constructor) instead of
   being a module-level global. That makes the pipeline unit-testable: tests
   feed it a tiny fake embedder with known vectors and assert on the result,
   with no 90 MB download. In production you inject the real SentenceTransformer.

2. `defend()` returns a DefenseResult that records what each stage removed.
   The UI uses this to *show* the filtering happening, stage by stage.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.cluster import DBSCAN

from . import config


# --- The embedder interface ----------------------------------------------
# Anything with an `.encode(list[str]) -> np.ndarray` method and an
# `is_degenerate` flag satisfies this. Defining it as a Protocol means we never
# have to import the heavy sentence-transformers library just to type-check.

class Embedder(Protocol):
    is_degenerate: bool
    def encode(self, texts: list[str]) -> np.ndarray: ...


class ZeroEmbedder:
    """Fallback used when the real embedding model fails to load.

    It returns zero vectors and flags itself as degenerate so the three
    embedding-based stages know to skip themselves (no-op) rather than make
    nonsense decisions. This is a small, deliberate improvement over the
    notebook, whose fallback could silently delete every comment.
    """
    is_degenerate = True

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 384), dtype=np.float32)


class SentenceTransformerEmbedder:
    """The real embedder used in production. Wraps sentence-transformers so the
    rest of the code only depends on the small `Embedder` interface above."""
    is_degenerate = False

    def __init__(self, model_name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts), dtype=np.float32)


def load_default_embedder() -> Embedder:
    """Try to load the real embedder; fall back to the safe zero embedder if the
    model can't be loaded (e.g. offline). The service stays up either way."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return ZeroEmbedder()


# --- Result objects -------------------------------------------------------

@dataclass
class StageTrace:
    """Record of what one filter stage did."""
    name: str
    removed: list[str] = field(default_factory=list)
    kept: int = 0


@dataclass
class DefenseResult:
    defended_code: str
    stages: list[StageTrace] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return sum(len(s.removed) for s in self.stages)


# --- Helpers --------------------------------------------------------------

def split_comments_and_code(code: str) -> tuple[list[str], list[str]]:
    """Partition lines into comment-like lines and everything else.

    A line is treated as a comment if, after stripping whitespace, it starts
    with //, /*, or #. (Same predicate as the notebook.)
    """
    comments, code_only = [], []
    for line in code.split("\n"):
        if line.strip().startswith(("//", "/*", "#")):
            comments.append(line)
        else:
            code_only.append(line)
    return comments, code_only


# --- The pipeline ---------------------------------------------------------

class CommentDefensePipeline:
    def __init__(self, embedder: Embedder | None = None):
        # If no embedder is supplied, default to the safe zero embedder so the
        # object always works. Production code injects the real one.
        self.embedder = embedder or ZeroEmbedder()

    # Each stage takes the current comment list (+ the code lines for context)
    # and returns (kept_comments, removed_comments). Keeping the signature
    # uniform is what lets `defend()` loop over them cleanly.

    def _stage_scead(self, comments, code_only):
        """Stage 1 — Semantic distance outlier filter.

        Embed every comment, find their centroid, drop the ones that sit far
        from it. A burst of off-topic injected comments tends to pull away
        from the cluster of any genuine comments.
        """
        if not comments or self.embedder.is_degenerate:
            return comments, []
        emb = self.embedder.encode(comments)
        centroid = emb.mean(axis=0)
        kept, removed = [], []
        for c, e in zip(comments, emb):
            (kept if np.linalg.norm(e - centroid) < config.SCEAD_DISTANCE_THRESHOLD
             else removed).append(c)
        return kept, removed

    def _stage_ccsa(self, comments, code_only):
        """Stage 2 — Comment/code similarity filter.

        Embed the code as one document, then drop any comment whose cosine
        similarity to the code is below threshold. Honest comments usually
        relate to the code; fake audit boilerplate does not.
        """
        if not comments or self.embedder.is_degenerate:
            return comments, []
        code_vec = self.embedder.encode(["\n".join(code_only)])[0]
        comment_emb = self.embedder.encode(comments)
        kept, removed = [], []
        for c, e in zip(comments, comment_emb):
            denom = np.linalg.norm(e) * np.linalg.norm(code_vec) + 1e-12
            sim = float(np.dot(e, code_vec) / denom)
            (kept if sim >= config.CCSA_SIMILARITY_THRESHOLD else removed).append(c)
        return kept, removed

    def _stage_topic(self, comments, code_only):
        """Stage 3 — Topic alignment: drop comments containing security-
        assurance keywords (audit, cert, codeql, verified, ...)."""
        kept, removed = [], []
        for c in comments:
            low = c.lower()
            (removed if any(w in low for w in config.TOPIC_SUSPICIOUS_WORDS)
             else kept).append(c)
        return kept, removed

    def _stage_cluster(self, comments, code_only):
        """Stage 4 — DBSCAN clustering: drop comments DBSCAN labels as noise
        (label -1). Injected comments that don't group with anything are
        treated as outliers."""
        if len(comments) <= 1 or self.embedder.is_degenerate:
            return comments, []
        emb = self.embedder.encode(comments)
        try:
            labels = DBSCAN(eps=config.DBSCAN_EPS,
                            min_samples=config.DBSCAN_MIN_SAMPLES).fit(emb).labels_
        except Exception:
            return comments, []  # clustering failed -> keep everything
        kept = [c for c, l in zip(comments, labels) if l != -1]
        removed = [c for c, l in zip(comments, labels) if l == -1]
        return kept, removed

    def _stage_ast(self, comments, code_only):
        """Stage 5 — AST consistency.

        KNOWN LIMITATION: this parses the code with Python's `ast` module, but
        the dataset is C/C++. `ast.parse` raises on C, we hit the except, and
        the stage becomes a no-op on virtually every real sample. We keep the
        original behavior for faithfulness and document it here. The proper fix
        is a language-aware parser such as tree-sitter (see README roadmap).
        """
        try:
            tree = ast.parse("\n".join(code_only))
        except Exception:
            return comments, []  # cannot parse (e.g. C/C++) -> no-op
        unsafe = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        kept, removed = [], []
        for c in comments:
            low = c.lower()
            claims_safety = ("sanit" in low or "validate" in low)
            has_unsafe_call = bool(unsafe & {"eval", "exec", "open"})
            (removed if claims_safety and has_unsafe_call else kept).append(c)
        return kept, removed

    def _stage_position(self, comments, code_only):
        """Stage 6 — Position risk: keep only comments in the central band of
        the file. Attackers love to bracket code with top/bottom banners;
        genuine comments tend to sit inline."""
        n = max(1, len(comments) + len(code_only))
        lo, hi = n * config.POSITION_RISK_PORTION, n * (1 - config.POSITION_RISK_PORTION)
        kept, removed = [], []
        # After earlier stages, comments occupy the first positions, so a
        # comment's index equals its position in the comment list.
        for i, c in enumerate(comments):
            (kept if lo <= i <= hi else removed).append(c)
        return kept, removed

    def _stage_lexical(self, comments, code_only):
        """Stage 7 — Lexical deception: a second, broader keyword sweep
        (safe, clean, pass, sanitiz, ...)."""
        kept, removed = [], []
        for c in comments:
            low = c.lower()
            (removed if any(w in low for w in config.LEXICAL_SUSPICIOUS_WORDS)
             else kept).append(c)
        return kept, removed

    def _stage_temporal(self, comments, code_only):
        """Stage 8 — Temporal consistency: drop comments carrying BOTH a year
        (19xx/20xx) and a version string (vN.N), a common metadata-spoof shape."""
        year_re, ver_re = re.compile(r"(19|20)\d{2}"), re.compile(r"v\d+\.\d+")
        kept, removed = [], []
        for c in comments:
            (removed if year_re.findall(c) and ver_re.findall(c) else kept).append(c)
        return kept, removed

    # The public entry point.
    def defend(self, code: str) -> DefenseResult:
        comments, code_only = split_comments_and_code(code)
        stages = [
            ("SCEAD (semantic distance)", self._stage_scead),
            ("CCSA (code similarity)", self._stage_ccsa),
            ("Topic alignment", self._stage_topic),
            ("DBSCAN clustering", self._stage_cluster),
            ("AST consistency", self._stage_ast),
            ("Position risk", self._stage_position),
            ("Lexical deception", self._stage_lexical),
            ("Temporal consistency", self._stage_temporal),
        ]
        trace: list[StageTrace] = []
        for name, fn in stages:
            comments, removed = fn(comments, code_only)
            trace.append(StageTrace(name=name, removed=removed, kept=len(comments)))

        defended = "\n".join(comments + code_only)
        return DefenseResult(defended_code=defended, stages=trace)
