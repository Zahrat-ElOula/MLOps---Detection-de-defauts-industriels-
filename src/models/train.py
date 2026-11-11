"""Training entry point with MLflow tracking.

This module implements the full training loop (progressive fine-tuning,
early stopping, checkpointing) and logs parameters, metrics and artefacts
to MLflow. The number of classes is inferred from the MVTec test
sub-directories and persisted to ``model_registry/classes.json``.

The test split is deterministic and stratified.  Thirty percent of the test
files are written to ``model_registry/split.json`` as the held-out set and are
never used for training or validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from src.config import Config, PROJECT_ROOT
from src.data.split import (
    HOLDOUT_FRACTION,
    SPLIT_SEED,
    VAL_FRACTION,
    build_split_manifest,
    relative_path,
    stratified_three_way_split,
    write_split_manifest,
)

MODEL_REGISTRY_DIR = PROJECT_ROOT / "model_registry"


def train(
    config,
    output_dir: Path | None = None,
    experiment_name: str | None = None,
    split_path: Path | None = None,
) -> dict:
    """Boucle d'entraînement complète avec MLflow."""
    import mlflow
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, ConcatDataset, Subset
    from src.data.dataset import MVTecDataset, discover_classes
    from src.data.augmentations import get_train_transforms, get_val_transforms
    from src.models.build import build_model
    from src.models.evaluate import compute_classification_metrics
    from src.utils.logger import get_logger

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    output_dir = Path(output_dir) if output_dir is not None else MODEL_REGISTRY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adapter l'API Config (mapping) à l'accès par attributs utilisé ci-dessous.
    if isinstance(config, Config):
        config = SimpleNamespace(**dict(config.params))
    elif isinstance(config, dict):
        config = SimpleNamespace(**dict(config))

    missing = [k for k in ("data_root", "category", "model_name", "lr",
                           "batch_size", "epochs", "image_size",
                           "early_stopping_patience")
               if not hasattr(config, k)]
    if missing:
        raise ValueError(f"Missing config keys: {missing}. "
                         f"Ajoutez-les dans configs/params.yaml et dans Config.")

    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(config.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    classes = discover_classes(data_root, config.category)
    num_classes = len(classes)
    logger.info(f"Classes découvertes ({num_classes}) : {classes}")

    # 1. Datasets
    # Split déterministe : 30 % du test est mis de côté (held-out) et n'est
    # utilisé ni pour l'entraînement ni pour la validation.  Le reste du test
    # est réparti entre train et validation.
    good_train = MVTecDataset(
        root=data_root, category=config.category,
        split="train", transform=get_train_transforms(config.image_size),
        classes=classes,
    )
    test_full = MVTecDataset(
        root=data_root, category=config.category,
        split="test", transform=get_val_transforms(config.image_size),
        classes=classes,
    )
    test_labels = list(test_full.labels)
    train_idx, val_idx, heldout_idx = stratified_three_way_split(
        test_labels,
        seed=SPLIT_SEED,
        holdout_fraction=HOLDOUT_FRACTION,
        val_fraction=VAL_FRACTION,
    )
    if not val_idx and train_idx:
        # Très petits jeux de données : garder au moins un échantillon de
        # validation, sans jamais toucher au held-out.
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    logger.info(
        "Split déterministe (seed=%d) : train=%d val=%d held-out=%d (jamais vus)",
        SPLIT_SEED,
        len(train_idx),
        len(val_idx),
        len(heldout_idx),
    )

    split_manifest = build_split_manifest(
        test_full.samples,
        test_labels,
        classes,
        train_idx,
        val_idx,
        heldout_idx,
        seed=SPLIT_SEED,
        holdout_fraction=HOLDOUT_FRACTION,
        val_fraction=VAL_FRACTION,
    )
    split_file = write_split_manifest(
        split_manifest, split_path or (output_dir / "split.json")
    )
    logger.info("Split manifest écrit : %s", split_file)

    train_ds = ConcatDataset([good_train, Subset(test_full, train_idx)])
    val_ds = Subset(test_full, val_idx)
    train_labels = list(good_train.labels) + [test_labels[i] for i in train_idx]

    # 2. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 3. Modèle, loss pondérée, optim, scheduler
    model = build_model(config.model_name, num_classes=num_classes, pretrained=True).to(device)
    # --- Fine-tuning progressif : gel du backbone ---
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    unfreeze_epoch = 5
    logger.info(f"Backbone gelé. Dégel prévu à l'epoch {unfreeze_epoch}.")

    counts = np.bincount(np.asarray(train_labels), minlength=num_classes).astype(float)
    class_weights = len(train_labels) / (num_classes * np.maximum(counts, 1.0))
    logger.info(f"Pondération des classes : {class_weights.tolist()}")
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    classes_path = output_dir / "classes.json"
    classes_path.write_text(json.dumps(list(classes), indent=2), encoding="utf-8")

    # 4. MLflow
    mlflow.set_experiment(experiment_name or "mvtec-bottle-classification")
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config.model_name,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "image_size": config.image_size,
            "early_stopping_patience": config.early_stopping_patience,
            "category": config.category,
            "num_classes": num_classes,
            "classes": ",".join(classes),
            "num_train": len(train_loader.dataset),
            "num_val": len(val_loader.dataset),
            "num_held_out": len(heldout_idx),
            "split_seed": SPLIT_SEED,
        })

        best_val_loss = float("inf")
        patience_counter = 0
        metrics: dict[str, Any] = {}
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(config.epochs):
            if epoch == unfreeze_epoch:
                # Dégeler tous les paramètres
                for param in model.parameters():
                    param.requires_grad = True
                # RECRÉER l'optimiseur pour inclure les paramètres dégelés
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr / 10)
                # Recréer le scheduler pour le nouvel optimiseur
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
                logger.info(f"Epoch {epoch+1} : dégel complet, nouvel optimiseur avec LR={config.lr/10}.")
            # --- train ---
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_loader.dataset)

            # --- val ---
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            val_loss /= len(val_loader.dataset)

            # --- métriques multi-classes ---
            metrics = compute_classification_metrics(
                all_labels, all_preds, average="macro" if num_classes > 2 else None
            )
            logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                        f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['f1']:.4f}")

            # --- MLflow logging ---
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_acc": metrics["accuracy"], "val_f1": metrics["f1"]}, step=epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(metrics["accuracy"])
            history["val_f1"].append(metrics["f1"])

            # --- scheduler + early stopping ---
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "model.pt")
                logger.info(f"  → best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping à l'epoch {epoch+1}")
                    break

        # 5. Sauvegarder métriques finales + artefacts
        payload = {
            "best_val_loss": best_val_loss,
            "classes": list(classes),
            "num_classes": num_classes,
            "class_weights": class_weights.tolist(),
            "split_path": relative_path(split_file),
            "split_seed": SPLIT_SEED,
            "num_held_out": len(heldout_idx),
            **metrics,
            "history": history,
        }
        (output_dir / "train_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(output_dir / "train_metrics.json"))
        mlflow.log_artifact(str(classes_path))
        mlflow.log_artifact(str(split_file))
        mlflow.log_artifact(str(output_dir / "model.pt"))

    return payload
