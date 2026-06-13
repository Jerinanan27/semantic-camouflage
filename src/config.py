"""
Central configuration.

Why this file exists
--------------------
In your notebook the model names and the defense thresholds (1.20, 0.15, 0.8,
0.2, ...) were sprinkled through the function bodies. That is fine for a
one-off experiment, but in a service you want a single place to look when you
ask "what value is stage 1 using?" or "which model are we serving?".

Everything here can also be overridden by an environment variable, which is
how you configure a container in production without editing code. Example:

    DETECTOR_MODEL=myuser/graphcodebert-vuln uvicorn src.api:app
"""

import os


# --- Models ---------------------------------------------------------------

# The fine-tuned vulnerability detector. Point this at YOUR Hugging Face repo
# once you upload your checkpoint. Until then it falls back to the base model,
# which lets the app boot but will NOT give meaningful predictions (the
# classification head is untrained). The API exposes whether a real checkpoint
# was loaded so the UI can warn the user.
DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "Jerin27/graphcodebert-vuln")

# The model that generates the LLM-augmented camouflage comments.
ATTACK_MODEL = os.getenv("ATTACK_MODEL", "google/flan-t5-base")

# The sentence-embedding model the defense uses to measure how "off-topic" a
# comment is relative to the code.
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Max tokens the detector reads from a function (matches your training run).
MAX_LEN = int(os.getenv("MAX_LEN", "512"))


# --- Defense thresholds ---------------------------------------------------
# These are lifted verbatim from your notebook so behavior matches the paper.

SCEAD_DISTANCE_THRESHOLD = 1.20      # stage 1: drop comments far from the centroid
CCSA_SIMILARITY_THRESHOLD = 0.15     # stage 2: drop comments dissimilar to the code
DBSCAN_EPS = 0.8                     # stage 4: clustering radius
DBSCAN_MIN_SAMPLES = 2               # stage 4: min points to form a cluster
POSITION_RISK_PORTION = 0.2          # stage 6: keep only the central 60% of lines

# Keyword lists used by the lexical / topic filters (stages 3 and 7).
TOPIC_SUSPICIOUS_WORDS = [
    "audit", "cert", "codeql", "oss-fuzz", "validated", "verified",
    "patched", "100%", "guarantee", "secure",
]
LEXICAL_SUSPICIOUS_WORDS = [
    "cert", "codeql", "oss-fuzz", "audit", "validated", "verified",
    "safe", "clean", "pass", "patched", "sanitiz", "ai_gen", "ai-gen",
]
