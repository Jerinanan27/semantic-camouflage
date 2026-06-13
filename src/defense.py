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

    Single-line comments start (after stripping) with //, /*, or #. Multi-line
    /* ... */ blocks are tracked so that EVERY line inside the block — not just
    the opening line — is classified as a comment. This closes the "block
    comment leak": previously the interior lines of an injected /* audit
    summary */ block were mistaken for code and survived the defense.
    """
    comments, code_only = [], []
    in_block = False
    for line in code.split("\n"):
        stripped = line.strip()
        if in_block:
            comments.append(line)
            if "*/" in line:               # block ends on this line
                in_block = False
        elif stripped.startswith(("//", "#")):
            comments.append(line)
        elif stripped.startswith("/*"):
            comments.append(line)
            if "*/" not in stripped:        # block stays open past this line
                in_block = True
        else:
            code_only.append(line)
    return comments, code_only


# --- Code parsing for the AST stage --------------------------------------

# Dangerous functions whose presence makes a "sanitized / validated" claim in a
# nearby comment suspicious. Now that we can parse C/C++, this includes the
# classic unsafe C calls, not just Python's eval/exec/open.
_UNSAFE_CALLS = {"eval", "exec", "open", "system", "popen",
                 "strcpy", "strcat", "sprintf", "gets", "scanf",
                 "memcpy", "malloc", "free", "realloc"}

_c_parser = None
_c_parser_tried = False


def _get_c_parser():
    """Lazily build a tree-sitter C parser, once. Returns None if tree-sitter
    isn't available, so the AST stage degrades gracefully instead of crashing."""
    global _c_parser, _c_parser_tried
    if _c_parser_tried:
        return _c_parser
    _c_parser_tried = True
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_c
        _c_parser = Parser(Language(tree_sitter_c.language()))
    except Exception:
        _c_parser = None
    return _c_parser


def extract_call_names(code: str) -> set[str]:
    """Return the set of function names called in `code`.

    Tries tree-sitter (handles C/C++) first, then falls back to Python's `ast`
    (handles Python), then to an empty set. This is the fix for the old AST
    stage, which used Python's `ast` on C code and silently did nothing.
    """
    parser = _get_c_parser()
    if parser is not None:
        try:
            tree = parser.parse(bytes(code, "utf8"))
            names: set[str] = set()
            src = code.encode("utf8")

            def walk(node):
                if node.type == "call_expression":
                    fn = node.child_by_field_name("function")
                    if fn is not None and fn.type == "identifier":
                        names.add(src[fn.start_byte:fn.end_byte].decode("utf8"))
                for child in node.children:
                    walk(child)

            walk(tree.root_node)
            if names:
                return names
        except Exception:
            pass

    # Fallback: Python source.
    try:
        import ast as _ast
        tree = _ast.parse(code)
        return {n.func.id for n in _ast.walk(tree)
                if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
    except Exception:
        return set()


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
        kept = [c for c, lbl in zip(comments, labels) if lbl != -1]
        removed = [c for c, lbl in zip(comments, labels) if lbl == -1]
        return kept, removed

    def _stage_ast(self, comments, code_only):
        """Stage 5 — AST consistency (now C/C++ aware).

        Parses the executable lines and extracts the function calls. A comment
        that *claims* the code is sanitized/validated is removed when the code
        actually calls a dangerous function (strcpy, gets, system, eval, ...) —
        the claim contradicts the code. Previously this used Python's `ast` on
        C code and was a silent no-op; it now uses tree-sitter (see
        `extract_call_names`).
        """
        unsafe = extract_call_names("\n".join(code_only)) & _UNSAFE_CALLS
        if not unsafe:
            return comments, []
        kept, removed = [], []
        for c in comments:
            low = c.lower()
            claims_safety = ("sanit" in low or "validate" in low or "safe" in low)
            (removed if claims_safety else kept).append(c)
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
