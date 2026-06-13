---
title: Semantic Camouflage
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Semantic Camouflage — Attacking and Defending a Code Vulnerability Detector

**[▶ Live demo](https://huggingface.co/spaces/Jerin27/semantic-camouflage)** · **[Model](https://huggingface.co/Jerin27/graphcodebert-vuln)** · **[Code](https://github.com/Jerinanan27/semantic-camouflage)**

An AI model (GraphCodeBERT, fine-tuned) reads a C/C++ function and predicts
whether it is vulnerable. This project demonstrates — live and interactively —
a weakness in such models and a defense against it:

1. **It works.** The detector flags real vulnerabilities (≈0.88 F1 on held-out data).
2. **It can be fooled.** Wrapping the *unchanged* code in fake "all-clear"
   comments — `// CodeQL scan: no issues detected`, fabricated audit summaries —
   collapses the model's confidence and can flip its verdict to "safe."
3. **It can be defended.** An 8-stage filter strips the deceptive comments and
   the correct verdict returns.

This is a deployed reproduction and extension of a CSE487 research project,
itself built on the FuncVul dataset (Halder et al., 2025).

---

## How it works

```
  code ──▶ /predict ──▶ verdict + confidence
       └─▶ /attack  ──▶ camouflaged code + (often flipped) verdict
       └─▶ /defend  ──▶ filtered code + restored verdict + per-stage trace
       └─▶ /pipeline ─▶ all three stages in one call (powers the demo)
```

- **Detector** (`src/detector.py`) — fine-tuned GraphCodeBERT, served from the
  Hugging Face Hub.
- **Attack** (`src/attack.py`) — heuristic (deterministic templates) and
  LLM-augmented (FLAN-T5) comment injection. Executable code is never altered.
- **Defense** (`src/defense.py`) — an 8-stage pipeline: embedding-distance
  outlier removal, code/comment similarity, keyword and lexical filters, DBSCAN
  clustering, AST checks, positional and temporal heuristics.
- **API** (`src/api.py`) — FastAPI service, containerized with Docker, deployed
  on Hugging Face Spaces.

---

## Results

**Detector quality (held-out test set, grouped by CVE to prevent leakage):**

| Metric | Score |
|---|---|
| F1 | 0.879 |
| Accuracy | 0.889 |
| Recall | 0.931 |
| False-negative rate | 0.069 |

These match the source study's clean GraphCodeBERT baseline (F1 ≈ 0.875).

**Attack effectiveness** (35 genuinely-vulnerable snippets; "flip" = verdict
changed from vulnerable to safe):

| Attack | Verdict flipped | Avg. confidence drop |
|---|---|---|
| Heuristic | 13 / 35 (37%) | −0.40 |
| LLM-augmented | 10 / 35 (29%) | −0.35 |

On samples that did not fully flip, confidence still fell sharply (e.g. a
use-after-free bug dropped from 0.98 → 0.20 under attack, then recovered to 0.98
after defense).

**An honest divergence from the source paper.** The original study reports the
LLM-augmented attack as *stronger* than the heuristic one. In this deployed
reproduction, on short function snippets, the **heuristic attack was slightly
stronger** — the FLAN-T5-generated comments were often too generic to move the
model as much as blunt, high-signal banners. This is reported as found rather
than tuned to match the paper.

---

## Quick start (local)

```bash
pip install -r requirements.txt
pytest                          # 24 tests
uvicorn src.api:app --reload    # then open http://127.0.0.1:8000
```

Re-train the detector reproducibly (designed for Kaggle/Colab with a GPU):

```bash
python scripts/train.py --push-to-hub YOURNAME/graphcodebert-vuln
```

---

## Tech stack

Python · PyTorch · Hugging Face Transformers · GraphCodeBERT · FLAN-T5 ·
sentence-transformers · scikit-learn · FastAPI · Docker · Hugging Face Spaces ·
pytest.

---

## Known limitations

Stated plainly, because they are part of the analysis:

- **AST stage is a no-op on C/C++.** Stage 5 parses code with Python's `ast`
  module, which cannot parse C/C++, so the stage silently does nothing on the
  real dataset. Documented in `src/defense.py`. *Fix on roadmap: `tree-sitter`.*
- **Block comments leak.** The defense detects comments line-by-line, so the
  interior lines of multi-line `/* ... */` blocks are mistaken for code and
  survive. Verdicts still recover, but the defended text is imperfect.
- **Scope.** One dataset (FuncVul Dataset 1), function-level, four model
  architectures evaluated in the source study (only GraphCodeBERT is served).

---

## Project status

- [x] Reproducible training (matches paper's clean baseline)
- [x] Detector, attack, and defense modules — 24 tests passing
- [x] FastAPI service + interactive web demo
- [x] Deployed publicly on Hugging Face Spaces
- [X] CI/CD: auto-test and auto-deploy on push
- [x] CPU-only image slim-down
- [X] tree-sitter AST stage; block-comment-aware filtering

---

*Built on FuncVul (Halder, Ahmed & Camtepe, ESORICS 2025) and a CSE487 project,
"Semantic Camouflage: LLM-Augmented Evasion and Multi-Stage Defenses."*
