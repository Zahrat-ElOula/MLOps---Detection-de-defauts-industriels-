"""Albumentations pipelines used by training and evaluation.

Keeping augmentations in one module guarantees that validation and serving
preprocessing stay identical to what the model saw during training.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    Parameters
    ----------
    image_size:
        Output square resolution.

    Returns
    -------
    albumentations.Compose
        Geometric and photometric augmentations followed by ImageNet
        normalisation. Masks are returned as an additional ``mask`` key when
        supplied to the pipeline.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Build the deterministic validation/serving preprocessing pipeline.

    Parameters
    ----------
    image_size:
        Output square resolution.

    Returns
    -------
    albumentations.Compose
        Resize + normalisation only.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
