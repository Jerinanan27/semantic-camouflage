"""
train.py — Reproducibly fine-tune GraphCodeBERT for vulnerability detection.

WHAT THIS DOES
--------------
1. Loads FuncVul Dataset 1 (line-delimited JSON).
2. Splits it into train/val/test *grouped by CVE* so the same vulnerability
   never appears in two splits (prevents leaked, inflated scores).
3. Tokenizes with the GraphCodeBERT tokenizer.
4. Fine-tunes microsoft/graphcodebert-base for binary classification.
5. Keeps the checkpoint with the best validation F1.
6. Prints clean-condition test metrics (should land near the paper's ~0.875 F1).
7. Saves the model locally and (optionally) pushes it to the Hugging Face Hub,
   which is what the deployed service will download at startup.

HOW TO RUN ON KAGGLE
--------------------
See the chat message that accompanies this file for click-by-click steps. In
short: upload the six Dataset_*.json files as a Kaggle Dataset, turn on a GPU,
paste this file into a notebook cell (or add it as a script), and run.

The data-loading functions at the top use only json/ast/pandas/sklearn, so they
can be unit-tested without downloading any model.
"""

import argparse
import ast
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# =========================================================================
# CONFIG — change these if needed. Defaults match the paper.
# =========================================================================
MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LEN = 512
NUM_LABELS = 2
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
LEARNING_RATE = 2e-5
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
SEED = 42
TEST_SIZE = 0.20      # 20% of all data -> test
VAL_SIZE = 0.10       # 10% of all data -> validation


# =========================================================================
# PART 1 — DATA (no heavy imports; testable on its own)
# =========================================================================

def parse_code_chunks(value) -> str:
    """Turn the `code_chunks` field into a single code string.

    The field is inconsistent across records:
      * sometimes a real list of code lines  -> join with newlines
      * sometimes a raw multiline string      -> use as-is
    Both cases produce clean, multiline code. (Faithful to the notebook.)
    """
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value)


def load_dataset_1(path: str) -> pd.DataFrame:
    """Read the line-delimited JSON file into a DataFrame with columns:
    text (the code), label (0/1), cve (used for the grouped split)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "code_chunks" not in obj or "label" not in obj:
                continue
            rows.append({
                "text": parse_code_chunks(obj["code_chunks"]).strip(),
                "label": int(obj["label"]),
                "cve": str(obj.get("cve", "unknown")),
            })
    if not rows:
        raise ValueError(f"No usable records found in {path}")
    df = pd.DataFrame(rows)
    df = df[df["text"].astype(bool)].reset_index(drop=True)   # drop empties
    return df


def grouped_split(df: pd.DataFrame, seed=SEED):
    """Split into train/val/test grouping by CVE, so no CVE spans two splits.

    Why grouping matters: if line A and line B come from the same CVE patch and
    one lands in train while the other lands in test, the model can 'cheat' by
    memorizing the vulnerability. Grouping forbids that, giving an honest score.
    """
    groups = df["cve"].to_numpy()

    # First carve off the test set.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=groups))
    df_test = df.iloc[test_idx].reset_index(drop=True)
    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)

    # Then split the remainder into train and validation.
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)   # val as a fraction of train+val
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval["cve"].to_numpy()))
    df_train = df_trainval.iloc[tr_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val, df_test


def assert_no_cve_leakage(df_train, df_val, df_test):
    """Safety check: confirm the three splits share no CVE. If this ever fails,
    the split is broken and the reported metrics would be untrustworthy."""
    s_tr, s_val, s_te = set(df_train.cve), set(df_val.cve), set(df_test.cve)
    assert not (s_tr & s_te), "LEAK: CVE in both train and test"
    assert not (s_tr & s_val), "LEAK: CVE in both train and val"
    assert not (s_val & s_te), "LEAK: CVE in both val and test"


# =========================================================================
# PART 2 — METRICS (numpy only; testable on its own)
# =========================================================================

def compute_metrics(eval_pred):
    """Return accuracy / precision / recall / f1 / fpr / fnr.

    The HF Trainer adds an 'eval_' prefix automatically, so we return the bare
    names here (returning 'eval_f1' would become 'eval_eval_f1').
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "fpr": fp / (fp + tn + eps),
        "fnr": fn / (fn + tp + eps),
    }


# =========================================================================
# PART 3 — TRAINING (heavy imports live INSIDE here on purpose, so importing
# this module for the data tests does not require torch/transformers)
# =========================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_dataset_path() -> str:
    """Locate Dataset_1.json. On Kaggle it lives under /kaggle/input/<your-ds>/.
    We search rather than hard-code so you don't have to edit the path."""
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn == "Dataset_1.json":
                candidates.append(os.path.join(root, fn))
    if candidates:
        return candidates[0]
    # Fallback: current directory (e.g. local testing).
    if os.path.exists("Dataset_1.json"):
        return "Dataset_1.json"
    raise FileNotFoundError(
        "Dataset_1.json not found. Upload the dataset and check the path under /kaggle/input."
    )


def to_hf_dataset(df, tokenizer):
    """Tokenize a DataFrame into a HuggingFace Dataset ready for the Trainer."""
    from datasets import Dataset
    ds = Dataset.from_dict({"text": df["text"].tolist(),
                            "labels": [int(x) for x in df["label"].tolist()]})

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length",
                         truncation=True, max_length=MAX_LEN)

    ds = ds.map(tok, batched=True, remove_columns=["text"], load_from_cache_file=False)
    ds.set_format("torch")
    return ds


def filter_short(df, tokenizer):
    """Drop samples that tokenize to <= 2 tokens (empty / junk), as in the paper."""
    keep = []
    for i, text in df["text"].items():
        try:
            if len(tokenizer(text, truncation=False)["input_ids"]) > 2:
                keep.append(i)
        except Exception:
            pass
    return df.loc[keep].reset_index(drop=True)


def main(push_to_hub_repo: str | None = None, output_dir="./funcvul-graphcodebert"):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    path = find_dataset_path()
    print(f"Loading data from: {path}")
    df = load_dataset_1(path)
    print(f"Loaded {len(df)} samples | vulnerable: {df.label.mean():.1%}")

    df_train, df_val, df_test = grouped_split(df)
    assert_no_cve_leakage(df_train, df_val, df_test)
    print(f"Split (grouped by CVE): train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df_train = filter_short(df_train, tokenizer)
    df_val = filter_short(df_val, tokenizer)
    df_test = filter_short(df_test, tokenizer)

    ds_train = to_hf_dataset(df_train, tokenizer)
    ds_val = to_hf_dataset(df_val, tokenizer)
    ds_test = to_hf_dataset(df_test, tokenizer)

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",            # evaluate at the end of every epoch
        save_strategy="epoch",            # ...and save a checkpoint then
        load_best_model_at_end=True,      # restore the best one when done
        metric_for_best_model="f1",       # "best" = highest validation F1
        greater_is_better=True,
        fp16=torch.cuda.is_available(),   # mixed precision = faster on GPU
        logging_steps=50,
        report_to="none",                 # no wandb pop-ups
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # --- Final, honest evaluation on the held-out test set ---
    print("\n=== CLEAN TEST-SET METRICS (compare to paper ~0.875 F1) ===")
    test_metrics = trainer.evaluate(ds_test)
    for k, v in test_metrics.items():
        if k.startswith("eval_") and isinstance(v, float):
            print(f"  {k.replace('eval_',''):10s}: {v:.4f}")

    # --- Save locally (Kaggle puts ./ under /kaggle/working, downloadable) ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved model + tokenizer to: {output_dir}")

    # --- Optional: push to the Hugging Face Hub (your model registry) ---
    if push_to_hub_repo:
        print(f"Pushing to Hugging Face Hub: {push_to_hub_repo}")
        model.push_to_hub(push_to_hub_repo)
        tokenizer.push_to_hub(push_to_hub_repo)
        print("Push complete. Set DETECTOR_MODEL to this repo name in the service.")

    return test_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--push-to-hub", default=None,
                   help="HF repo id, e.g. yourname/graphcodebert-vuln")
    p.add_argument("--output-dir", default="./funcvul-graphcodebert")
    args = p.parse_args()
    main(push_to_hub_repo=args.push_to_hub, output_dir=args.output_dir)
