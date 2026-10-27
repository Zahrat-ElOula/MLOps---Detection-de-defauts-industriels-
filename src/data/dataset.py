"""MVTec AD dataset adapter.

Expected layout (subset of the official dataset)::

    data/raw/
    └── bottle/
        ├── train/
        │   └── good/*.png
        ├── test/
        │   ├── good/*.png
        │   └── broken_large/*.png
        └── ground_truth/
            └── broken_large/*_mask.png

The class is deliberately a skeleton: indexing is implemented so the rest of
the project can reason about dataset sizes, while pixel decoding and tensor
conversion are left to the student/engineer (see ``__getitem__``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from torch.utils.data import Dataset

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
GOOD_LABEL = 0
DEFECT_LABEL = 1


@dataclass
class MVTecDataset(Dataset):
    """PyTorch dataset for one MVTec AD category and split.

    Parameters
    ----------
    root:
        Path to the dataset root (contains one directory per category).
    category:
        MVTec category name (``"bottle"``, ``"screw"``, ...). When omitted the
        whole ``root`` is scanned.
    split:
        ``"train"``, ``"test"`` or ``"validation"``.
    transform:
        Optional callable applied to the loaded sample (Albumentations
        ``Compose`` or torchvision transform).
    image_size:
        Target square size kept for bookkeeping and default transforms.

    Attributes
    ----------
    samples:
        Sorted list of indexed image paths.
    labels:
        Binary labels aligned with ``samples`` (0 = good, 1 = defect).
    """

    root: str | Path
    category: str | None = None
    split: str = "train"
    transform: Callable[[object], object] | None = None
    image_size: int = 224
    samples: list[Path] = field(init=False, default_factory=list)
    labels: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Resolve paths and index the matching image files."""
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        search_root = self.root / self.category if self.category else self.root
        search_root = search_root / self.split
        if not search_root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {search_root}")

        self.samples = sorted(
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.labels = [
            GOOD_LABEL
            if "good" in path.relative_to(search_root).parts
            else DEFECT_LABEL
            for path in self.samples
        ]

    def __len__(self) -> int:
        """Return the number of indexed images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        """Load one sample and return ``(image, label)``."""
        from PIL import Image
        import numpy as np
        import torch

        path = self.samples[index]
        label = self.labels[index]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            try:
                import albumentations as A
                if isinstance(self.transform, A.Compose):
                    image = self.transform(image=np.array(image))["image"]
                else:
                    image = self.transform(image)
            except ImportError:
                image = self.transform(image)

        if not torch.is_tensor(image):
            arr = np.asarray(image, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = arr.transpose(2, 0, 1)
            image = torch.from_numpy(arr) / 255.0

        return image, label

    def class_names(self) -> Sequence[str]:
        """Return the label names in index order."""
        return ("good", "defect")
