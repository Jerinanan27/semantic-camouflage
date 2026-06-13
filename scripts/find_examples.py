"""
find_examples.py — Curate demo examples by finding which snippets the
camouflage attack actually flips.

It runs a set of genuinely-vulnerable code snippets through the real pipeline:
    clean -> predict -> attack (heuristic AND/OR llm) -> predict -> defend -> predict
and records, for each, how far the attack moved the model's confidence and
whether the verdict flipped. The best flip cases are saved for the demo UI.

WHY THIS EXISTS
---------------
Not every snippet gets fooled (the paper's attack success rate is ~31-47%).
This script finds the ones that DO flip, so the live demo shows the effect
cleanly, and it produces honest summary stats you can quote.

Run from the project root:
    python scripts/find_examples.py --mode both --n 30
    python scripts/find_examples.py --mode heuristic --n 50   # fast, no FLAN-T5
"""

import argparse
import json
import os
import sys

# Make `src` importable when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import VulnerabilityDetector       # noqa: E402
from src.attack import Attacker                        # noqa: E402
from src.defense import CommentDefensePipeline, load_default_embedder  # noqa: E402


# A few hand-picked classic C bugs (short, unambiguous, demo-friendly).
HANDPICKED = [
    ("strcpy overflow",
     "char buf[64];\nstrcpy(buf, user_input);\nreturn process(buf);"),
    ("gets unbounded",
     "char line[128];\ngets(line);\nparse(line);"),
    ("sprintf overflow",
     "char path[256];\nsprintf(path, \"/home/%s/file\", username);\nopen(path);"),
    ("format string",
     "void log_msg(char *u) {\n    printf(u);\n}"),
    ("integer overflow alloc",
     "int n = read_count();\nchar *p = malloc(n * size);\nmemcpy(p, src, n * size);"),
    ("use after free",
     "free(ptr);\nptr->next = head;\nreturn ptr;"),
]


def load_dataset_snippets(n: int, path_hint: str | None) -> list[tuple[str, str]]:
    """Pull `n` short, demo-friendly vulnerable samples from Dataset 1."""
    path = path_hint
    if not path:
        for cand in ("Dataset_1.json", "data/Dataset_1.json",
                     os.path.expanduser("~/Dataset_1.json")):
            if os.path.exists(cand):
                path = cand
                break
    if not path or not os.path.exists(path):
        print("  (Dataset_1.json not found — using hand-picked snippets only)")
        return []

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("label") != 1:
                continue
            cc = o["code_chunks"]
            code = "\n".join(cc) if isinstance(cc, list) else str(cc)
            nlines = len([ln for ln in code.split("\n") if ln.strip()])
            if 3 <= nlines <= 12 and len(code) < 600:
                out.append((o.get("cve", "dataset"), code))
            if len(out) >= n:
                break
    return out


def evaluate(detector, attacker, defense, label, code, modes):
    """Return a result dict for one snippet across the requested attack modes."""
    clean = detector.predict(code)
    row = {"label": label, "code": code,
           "clean_conf_vuln": round(clean.prob_vulnerable, 4),
           "clean_verdict": clean.label, "modes": {}}
    for use_llm in modes:
        name = "llm" if use_llm else "heuristic"
        injected = attacker.inject(code, use_llm=use_llm)
        inj_pred = detector.predict(injected)
        defended = defense.defend(injected)
        def_pred = detector.predict(defended.defended_code)
        row["modes"][name] = {
            "injected_conf_vuln": round(inj_pred.prob_vulnerable, 4),
            "confidence_drop": round(clean.prob_vulnerable - inj_pred.prob_vulnerable, 4),
            "attack_flipped": clean.is_vulnerable and not inj_pred.is_vulnerable,
            "defense_restored": clean.is_vulnerable == def_pred.is_vulnerable,
            "injected_code": injected,
            "defended_code": defended.defended_code,
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "llm", "heuristic"], default="both")
    ap.add_argument("--n", type=int, default=30, help="dataset snippets to test")
    ap.add_argument("--dataset", default=None, help="path to Dataset_1.json")
    ap.add_argument("--out", default="data/examples/demo_examples.json")
    args = ap.parse_args()

    modes = {"both": [False, True], "llm": [True], "heuristic": [False]}[args.mode]
    print(f"Attack modes: {'heuristic + llm' if len(modes) == 2 else ('llm' if modes[0] else 'heuristic')}")

    # Build the snippet set: hand-picked + dataset samples.
    snippets = list(HANDPICKED) + load_dataset_snippets(args.n, args.dataset)
    print(f"Testing {len(snippets)} snippets "
          f"({len(HANDPICKED)} hand-picked + {len(snippets) - len(HANDPICKED)} from Dataset 1)\n")

    # Load models ONCE (this is why we import directly instead of using the API).
    print("Loading models (first run downloads them; please wait)...")
    detector = VulnerabilityDetector()
    detector.load()
    attacker = Attacker(rng_seed=42)   # seeded -> reproducible examples
    defense = CommentDefensePipeline(embedder=load_default_embedder())
    if True in modes:
        print("Warming up FLAN-T5 (the ~1GB attack model)...")
        attacker._load_t5()
    print("Ready.\n")

    results = []
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = f"{'#':>3}  {'snippet':28}  {'clean':>6}  {'mode':9}  {'inj':>6}  {'drop':>6}  flip?"
    print(header)
    print("-" * len(header))

    for i, (label, code) in enumerate(snippets, 1):
        try:
            row = evaluate(detector, attacker, defense, label, code, modes)
        except Exception as e:
            print(f"{i:>3}  {label[:28]:28}  ERROR: {e}")
            continue
        results.append(row)
        for mname, m in row["modes"].items():
            flip = "FLIP" if m["attack_flipped"] else "."
            print(f"{i:>3}  {label[:28]:28}  {row['clean_conf_vuln']:.3f}  "
                  f"{mname:9}  {m['injected_conf_vuln']:.3f}  {m['confidence_drop']:+.3f}  {flip}")
        # Save incrementally so an interrupt never loses progress.
        json.dump(results, open(args.out, "w"), indent=2)

    # --- Summary stats (the honest numbers you can quote) ---
    print("\n" + "=" * 60)
    for use_llm in modes:
        mname = "llm" if use_llm else "heuristic"
        rel = [r for r in results if r["clean_verdict"] == "vulnerable"]
        flips = sum(r["modes"][mname]["attack_flipped"] for r in rel)
        drops = [r["modes"][mname]["confidence_drop"] for r in rel]
        avg_drop = sum(drops) / len(drops) if drops else 0
        print(f"{mname:10s}: flipped {flips}/{len(rel)} "
              f"({flips / max(1, len(rel)):.0%})  |  avg confidence drop {avg_drop:+.3f}")

    # Keep only the clean flip cases for the demo UI, best drop first.
    best = []
    for r in results:
        for mname, m in r["modes"].items():
            if m["attack_flipped"] and m["defense_restored"]:
                best.append({"label": r["label"], "mode": mname,
                             "code": r["code"],
                             "clean_conf": r["clean_conf_vuln"],
                             "injected_conf": m["injected_conf_vuln"]})
    best.sort(key=lambda x: x["clean_conf"] - x["injected_conf"], reverse=True)
    json.dump({"results": results, "best_demo_examples": best},
              open(args.out, "w"), indent=2)
    print(f"\nSaved {len(best)} clean flip examples to {args.out}")


if __name__ == "__main__":
    main()
