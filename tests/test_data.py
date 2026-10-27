"""Tests for the MVTec dataset adapter and augmentations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import DEFECT_LABEL, GOOD_LABEL, MVTecDataset


@pytest.fixture()
def fake_mvtec_root(tmp_path: Path) -> Path:
    """Create a minimal MVTec-like directory tree with empty images."""
    good_dir = tmp_path / "bottle" / "train" / "good"
    defect_dir = tmp_path / "bottle" / "test" / "broken_large"
    good_dir.mkdir(parents=True)
    defect_dir.mkdir(parents=True)
    (good_dir / "000.png").write_bytes(b"not-an-image")
    (good_dir / "001.png").write_bytes(b"not-an-image")
    (defect_dir / "002.png").write_bytes(b"not-an-image")
    return tmp_path


def test_dataset_indexes_images_and_labels(fake_mvtec_root: Path) -> None:
    """The adapter indexes files and derives good/defect labels."""
    dataset = MVTecDataset(root=fake_mvtec_root, category="bottle", split="train")

    assert len(dataset) == 2
    assert dataset.labels == [GOOD_LABEL, GOOD_LABEL]
    assert dataset.class_names() == ("good", "defect")


def test_dataset_detects_defect_split(fake_mvtec_root: Path) -> None:
    """Files outside a ``good`` directory are labelled as defects."""
    dataset = MVTecDataset(root=fake_mvtec_root, category="bottle", split="test")

    assert len(dataset) == 1
    assert dataset.labels == [DEFECT_LABEL]


def test_dataset_missing_root_raises(tmp_path: Path) -> None:
    """A missing dataset root fails fast."""
    with pytest.raises(FileNotFoundError):
        MVTecDataset(root=tmp_path / "missing")


def test_getitem_loads_rgb_tensor(fake_mvtec_root: Path) -> None:
    """__getitem__ returns a (C, H, W) float32 tensor and a label."""
    from PIL import Image
    # Écrire une vraie image PNG dans le fixture (le fixture crée un fichier vide)
    img_path = fake_mvtec_root / "bottle" / "train" / "good" / "000.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

    from src.data.augmentations import get_val_transforms
    dataset = MVTecDataset(
        root=fake_mvtec_root, category="bottle", split="train",
        transform=get_val_transforms(224),
    )
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype.is_floating_point
    assert label == 0


def test_validation_transforms_produce_chw_tensor() -> None:
    """Validation preprocessing resizes and returns a CHW tensor."""
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    transformed = get_val_transforms(image_size=224)(image=image)["image"]

    assert tuple(transformed.shape) == (3, 224, 224)


def test_train_transforms_are_deterministic_in_shape() -> None:
    """Training augmentations keep the model input shape stable."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    transformed = get_train_transforms(image_size=224)(image=image)["image"]

    assert tuple(transformed.shape) == (3, 224, 224)
