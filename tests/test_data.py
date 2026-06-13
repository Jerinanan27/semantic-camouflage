"""
Tests for the pure helpers in scripts/train.py.

These cover the data parsing and metric math — the parts that don't need a GPU
or a model download, so CI can run them in seconds.
"""

import os
import sys

import numpy as np

# Make scripts/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train import parse_code_chunks, compute_metrics  # noqa: E402


def test_parse_code_chunks_handles_list():
    # A list of lines becomes newline-joined code.
    assert parse_code_chunks(["int x = 0;", "return x;"]) == "int x = 0;\nreturn x;"


def test_parse_code_chunks_handles_string():
    # A raw multiline string is passed through untouched.
    raw = "if (length < 2)\n\tgoto trunc;"
    assert parse_code_chunks(raw) == raw


def test_compute_metrics_perfect_predictions():
    logits = np.array([[2.0, -2.0], [-2.0, 2.0]])  # -> predicts [0, 1]
    labels = np.array([0, 1])
    m = compute_metrics((logits, labels))
    assert round(m["f1"], 3) == 1.0
    assert round(m["accuracy"], 3) == 1.0
    assert m["fpr"] < 1e-6 and m["fnr"] < 1e-6


def test_compute_metrics_all_wrong():
    logits = np.array([[-2.0, 2.0], [2.0, -2.0]])  # -> predicts [1, 0]
    labels = np.array([0, 1])
    m = compute_metrics((logits, labels))
    assert m["f1"] < 1e-6           # nothing correct
    assert round(m["fnr"], 3) == 1.0  # the one true vuln was missed
