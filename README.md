# Semantic Camouflage — Attacking and Defending a Code Vulnerability Detector

> Live demo: _(coming Day 3)_ · Model card: _(coming Day 1 homework)_

A transformer-based detector (**GraphCodeBERT**) decides whether a C/C++
function is vulnerable. This project shows three things, live and interactively:

1. **It works.** Paste a function, get a vulnerable / not-vulnerable verdict.
2. **It can be fooled.** Inject "semantic camouflage" — fake-trust comments like
   `// CodeQL scan: no issues detected` and LLM-generated audit summaries — and
   the prediction flips, *even though the executable code never changed*.
3. **It can be defended.** An 8-stage filter strips the deceptive comments and
   the correct prediction returns.

This is a deployment of a CSE487 research project; the underlying study compared
four models (GraphCodeBERT, CodeBERT, UniXcoder, BERT-base) under heuristic and
LLM-augmented comment-injection attacks. See `RESULTS.md` for the full table.

## Architecture

```
  Code in ──▶ /predict ──▶ verdict + confidence
          └─▶ /attack  ──▶ injected code + (flipped) verdict
          └─▶ /defend  ──▶ filtered code + (restored) verdict + per-stage trace
```

- **Detector:** fine-tuned GraphCodeBERT, served from the Hugging Face Hub.
- **Attack:** `src/attack.py` — heuristic + FLAN-T5 comment injection.
- **Defense:** `src/defense.py` — the 8-stage pipeline (this file is the centerpiece).
- **API:** FastAPI (`src/api.py`), containerized with Docker, deployed via CI/CD.

## Known limitations (read before the interview)

- **Stage 5 (AST consistency) is a no-op on C/C++.** It uses Python's `ast`
  parser; C code doesn't parse, so the stage silently does nothing on the real
  dataset. Documented in `src/defense.py`. Roadmap fix: a `tree-sitter` parser.
- Single dataset (FuncVul Dataset 1), function-level only.

## Quick start

```bash
pip install -r requirements.txt
pytest                       # run the test suite
uvicorn src.api:app --reload # start the API locally (added Day 2)
```

## Roadmap / status

- [x] Day 1 — repo scaffold, defense pipeline refactored + tested
- [x] Day 2 — detector + attack modules, FastAPI service (24 tests passing)
- [ ] Day 3 — frontend + public deploy
- [ ] Day 4 — CI/CD (GitHub Actions)
- [ ] Day 5 — metrics + observability
- [ ] Day 6 — README polish, demo GIF, model card
- [ ] Day 7 — buffer, demo video, talking points
