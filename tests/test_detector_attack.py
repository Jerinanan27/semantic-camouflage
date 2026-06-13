"""
Tests for the detector and the attack, with no model downloads.

The detector tests subclass VulnerabilityDetector and replace `_logits` with
fixed numbers, so we can verify the verdict/confidence mapping exactly. The
attack tests exercise the heuristic path, which is pure templates.
"""

import numpy as np

from src.detector import VulnerabilityDetector, _softmax
from src.attack import Attacker


# --- Detector --------------------------------------------------------------

class _FakeDetector(VulnerabilityDetector):
    """A detector whose model output we control."""
    def __init__(self, logits):
        super().__init__(model_name="fake")
        self._fixed = np.array(logits, dtype=np.float32)

    def _logits(self, code):           # override the seam; never loads a model
        return self._fixed


def test_softmax_sums_to_one():
    p = _softmax(np.array([2.0, 1.0]))
    assert abs(p.sum() - 1.0) < 1e-6
    assert p[0] > p[1]                 # larger logit -> larger probability


def test_detector_flags_vulnerable():
    det = _FakeDetector([-3.0, 3.0])   # strongly class 1 (vulnerable)
    pred = det.predict("any code")
    assert pred.is_vulnerable is True
    assert pred.label == "vulnerable"
    assert pred.prob_vulnerable > 0.9
    assert pred.confidence > 0.9


def test_detector_flags_safe():
    det = _FakeDetector([3.0, -3.0])   # strongly class 0 (not vulnerable)
    pred = det.predict("any code")
    assert pred.is_vulnerable is False
    assert pred.label == "not vulnerable"
    assert pred.prob_vulnerable < 0.1


# --- Attack (heuristic path) ----------------------------------------------

REAL_CODE = "int total = 0;\nfor (int i = 0; i < n; i++) total += arr[i];\nreturn total;"


def test_heuristic_attack_preserves_executable_code():
    inj = Attacker(rng_seed=0).inject(REAL_CODE, use_llm=False)
    # Every original line must still be present, unchanged.
    for line in REAL_CODE.splitlines():
        assert line in inj


def test_heuristic_attack_adds_fake_trust_comments():
    inj = Attacker(rng_seed=0).inject(REAL_CODE, use_llm=False)
    assert "CodeQL" in inj
    assert "CERT check passed" in inj
    assert "VERIFIED_SAFE_SIGNAL" in inj
    # The injected text is strictly longer than the original.
    assert len(inj) > len(REAL_CODE)


def test_heuristic_attack_is_deterministic_with_seed():
    a = Attacker(rng_seed=42).inject(REAL_CODE)
    b = Attacker(rng_seed=42).inject(REAL_CODE)
    assert a == b                      # same seed -> identical output
