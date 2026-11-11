"""Push a batch of user feedback images to trigger Continuous Training.

The script counts feedback images that are new or modified according to Git
and, when the batch is large enough, commits and pushes them. The push on
``data/feedback/**`` then triggers the CT workflow.

Usage
-----
    python scripts/auto_push_feedback.py --dry-run
    python scripts/auto_push_feedback.py --n 5
    FEEDBACK_BATCH_SIZE=5 python scripts/auto_push_feedback.py

The default batch size is 5 (``--n`` overrides ``FEEDBACK_BATCH_SIZE``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _pending_images() -> list[Path]:
    """Return untracked or modified feedback images."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--modified",
            "--exclude-standard",
            "--",
            "data/feedback",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    paths = [
        PROJECT_ROOT / line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]


def _run(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Commit and push a batch of feedback images."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Minimum number of new images (default: FEEDBACK_BATCH_SIZE or 5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without committing or pushing.",
    )
    args = parser.parse_args()

    batch_size = args.n or int(os.getenv("FEEDBACK_BATCH_SIZE", "5"))
    if batch_size < 1:
        print("Batch size must be >= 1.", file=sys.stderr)
        return 2

    if not FEEDBACK_DIR.is_dir():
        print(f"No feedback directory: {FEEDBACK_DIR}")
        return 0

    images = _pending_images()
    print(f"Pending feedback images: {len(images)} (threshold: {batch_size})")
    for image in images:
        print(f"  - {image.relative_to(PROJECT_ROOT)}")

    if len(images) < batch_size:
        print("Not enough new feedback images; nothing to push.")
        return 0

    message = f"feedback: {len(images)} new images [auto]"
    _run(["git", "add", "data/feedback"], args.dry_run)
    _run(["git", "commit", "-m", message], args.dry_run)
    _run(["git", "push", "origin", "main"], args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
