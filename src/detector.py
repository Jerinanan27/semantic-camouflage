"""
The vulnerability detector.

This wraps your fine-tuned GraphCodeBERT model behind one tiny method:
`predict(code) -> Prediction`. The web API and the demo never touch tokenizers
or tensors directly; they ask the detector for a verdict and get back a plain
object. That boundary is what keeps the rest of the system simple.

Two patterns worth noticing
---------------------------
1. LAZY LOADING. The 500 MB model is NOT loaded when you create the object — it
   loads on first use (or when you call .load()). This matters for a web
   service: the process can start instantly and report "alive" before the slow
   model download finishes, and tests can construct the object without a GPU.

2. A TESTABLE SEAM. All the model machinery funnels through one private method,
   `_logits(code)`. Tests override just that method to return fixed numbers,
   which lets us verify the verdict/confidence logic with no model at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Prediction:
    """A plain, JSON-friendly result object."""
    is_vulnerable: bool
    label: str                 # "vulnerable" or "not vulnerable"
    confidence: float          # probability of the predicted class, 0..1
    prob_vulnerable: float     # probability of the 'vulnerable' class, 0..1


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw model scores into probabilities that sum to 1."""
    z = logits - np.max(logits)          # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum()


class VulnerabilityDetector:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.DETECTOR_MODEL
        self._model = None
        self._tokenizer = None
        self._device = None

    # --- loading -----------------------------------------------------------

    def load(self):
        """Download + load the model and tokenizer. Safe to call repeatedly."""
        if self._model is not None:
            return
        import torch
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name).to(self._device)
        self._model.eval()        # inference mode: no gradient tracking

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # --- the testable seam -------------------------------------------------

    def _logits(self, code: str) -> np.ndarray:
        """Run the model and return the two raw class scores [not_vuln, vuln].
        Tests override this method; production code uses the real model."""
        import torch
        self.load()
        inputs = self._tokenizer(
            code, return_tensors="pt", truncation=True, max_length=config.MAX_LEN,
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.logits[0].detach().cpu().numpy()

    # --- the public interface ---------------------------------------------

    def predict(self, code: str) -> Prediction:
        """Classify a single code string."""
        probs = _softmax(self._logits(code))
        prob_vuln = float(probs[1])         # class 1 == vulnerable
        is_vuln = prob_vuln >= 0.5
        return Prediction(
            is_vulnerable=is_vuln,
            label="vulnerable" if is_vuln else "not vulnerable",
            confidence=float(max(probs)),
            prob_vulnerable=prob_vuln,
        )
