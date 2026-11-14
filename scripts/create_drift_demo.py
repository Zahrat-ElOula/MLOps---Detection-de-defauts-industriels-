"""Generate two demo folders for the drift monitoring demo.

- data/feedback_demo_normal : 20 healthy images from the SAME source as the
  reference (images 50-69; the monitoring script uses images 0-49 as reference)
  -> expected: no drift, no alert
- data/feedback_demo_drift  : the same 20 images with a strong brightness/contrast
  change -> expected: drift on all features, alert
"""

from pathlib import Path
import shutil

import cv2

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/raw/bottle/train/good"
DEST_NORMAL = ROOT / "data/feedback_demo_normal"
DEST_DRIFT = ROOT / "data/feedback_demo_drift"

REFERENCE_COUNT = 50   # must match limit=50 in monitoring_drift.py
DEMO_COUNT = 20
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


images = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in EXTENSIONS)
demo_images = images[REFERENCE_COUNT:REFERENCE_COUNT + DEMO_COUNT]

if len(demo_images) < DEMO_COUNT:
    raise RuntimeError(
        f"Need at least {REFERENCE_COUNT + DEMO_COUNT} source images, found {len(images)}."
    )

reset_dir(DEST_NORMAL)
reset_dir(DEST_DRIFT)

count = 0
for path in demo_images:
    img = cv2.imread(str(path))
    if img is None:
        continue

    # Normal: untouched copy
    cv2.imwrite(str(DEST_NORMAL / f"normal_{count:03d}.png"), img)

    # Drift: strong brightness and contrast change
    drifted = cv2.convertScaleAbs(img, alpha=1.8, beta=100)
    cv2.imwrite(str(DEST_DRIFT / f"drift_{count:03d}.png"), drifted)

    count += 1

print(f"{count} normal images -> {DEST_NORMAL}")
print(f"{count} drifted images -> {DEST_DRIFT}")