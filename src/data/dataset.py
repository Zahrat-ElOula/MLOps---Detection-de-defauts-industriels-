"""MVTec AD dataset adapter.

Expected layout (subset of the official dataset)::

    data/raw/
    └── bottle/
        ├── train/
        │   └── good/*.png
        ├── test/
        │   ├── good/*.png
        │   ├── broken_large/*.png
        │   └── broken_small/*.png
        └── ground_truth/
            └── broken_large/*_mask.png

The adapter indexes image files and derives labels from the sub-directory
name under the split folder (``test/broken_large`` -> ``broken_large``).
When ``classes`` is omitted, the legacy binary mapping is kept
(``good = 0``, any other folder = ``1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from torch.utils.data import Dataset

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
GOOD_LABEL = 0
DEFECT_LABEL = 1
DEFAULT_CLASSES: tuple[str, ...] = ("good", "defect")


def discover_classes(
    root: str | Path,
    category: str | None = None,
    split: str = "test",
) -> list[str]:
    """Discover the class names from the split sub-directories.

    ``good`` is always placed first so that index ``0`` keeps the historical
    meaning. Defect types are sorted alphabetically for reproducibility.
    """
    base = Path(root)
    if category:
        base = base / category
    split_dir = base / split
    if not split_dir.is_dir():
        return list(DEFAULT_CLASSES)

    names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if "good" in names:
        names.remove("good")
        names.insert(0, "good")
    if len(names) < 2:
        return list(DEFAULT_CLASSES)
    return names


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
    classes:
        Explicit class list (index order). When omitted, the legacy binary
        mapping is used.

    Attributes
    ----------
    samples:
        Sorted list of indexed image paths.
    labels:
        Integer labels aligned with ``samples``.
    """

    root: str | Path
    category: str | None = None
    split: str = "train"
    transform: Callable[[object], object] | None = None
    image_size: int = 224
    classes: Sequence[str] | None = None
    samples: list[Path] = field(init=False, default_factory=list)
    labels: list[int] = field(init=False, default_factory=list)
    _label_map: dict[str, int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Resolve paths and index the matching image files."""
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if self.classes is None:
            self.classes = DEFAULT_CLASSES
            self._label_map = None
        else:
            self.classes = tuple(self.classes)
            self._label_map = {name: index for index, name in enumerate(self.classes)}

        search_root = self.root / self.category if self.category else self.root
        search_root = search_root / self.split
        if not search_root.is_dir():
            raise FileNotFoundError(f"Split directory not found: {search_root}")

        self.samples = sorted(
            path
            for path in search_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.labels = [self._label_for(path, search_root) for path in self.samples]

    def _label_for(self, path: Path, search_root: Path) -> int:
        """Return the integer label for one file."""
        relative_parts = path.relative_to(search_root).parts
        folder = relative_parts[0] if len(relative_parts) > 1 else path.parent.name

        if self._label_map is None:
            return GOOD_LABEL if folder == "good" else DEFECT_LABEL

        if folder in self._label_map:
            return self._label_map[folder]

        # Flat files or unknown folders: keep "good" when the path contains it,
        # otherwise fall back to the first defect class.
        if "good" in relative_parts:
            return self._label_map.get("good", GOOD_LABEL)
        defect_names = [name for name in self.classes if name != "good"]
        if defect_names:
            return self._label_map[defect_names[0]]
        return self._label_map.get("defect", DEFECT_LABEL)

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
        return tuple(self.classes)
