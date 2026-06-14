"""Push the project to the Hugging Face Space using the official uploader."""

import os

from huggingface_hub import HfApi

REPO_ID = "Jerin27/semantic-camouflage"

token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    raise SystemExit("HF_TOKEN is empty. Set it as a repository secret on GitHub.")

api = HfApi(token=token)
api.upload_folder(
    folder_path=".",
    repo_id=REPO_ID,
    repo_type="space",
    commit_message="Auto-deploy from GitHub Actions",
    ignore_patterns=[
        ".git*", ".github/*", "tests/*", "scripts/*", "data/*",
        "**/__pycache__/*", ".pytest_cache/*", ".ruff_cache/*",
        "*.json", "requirements-dev.txt",
    ],
)
print(f"Uploaded to Space: https://huggingface.co/spaces/{REPO_ID}")
