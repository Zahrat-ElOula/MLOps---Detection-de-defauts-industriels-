"""Training entry point with MLflow tracking.

This module is a skeleton: the contract (signature, configuration keys,
artefacts) is fixed, the training loop is intentionally left to implement.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from src.config import Config, PROJECT_ROOT

MODEL_REGISTRY_DIR = PROJECT_ROOT / "model_registry"


def train(config, output_dir: Path | None = None,
          experiment_name: str | None = None) -> dict:
    """Boucle d'entraînement complète avec MLflow."""
    import mlflow
    import torch
    from torch.utils.data import DataLoader, ConcatDataset, random_split
    from src.data.dataset import MVTecDataset
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

    # 1. Datasets
    # MVP supervisé : train = train/good + 70% de test/ (good + defect).
    # Val = 30% de test/. À remplacer par un split stratifié (sklearn
    # StratifiedShuffleSplit) avant la soutenance pour garantir la présence
    # des deux classes en val.
    good_train = MVTecDataset(
        root=config.data_root, category=config.category,
        split="train", transform=get_train_transforms(config.image_size),
    )
    test_full = MVTecDataset(
        root=config.data_root, category=config.category,
        split="test", transform=get_val_transforms(config.image_size),
    )
    n_test = len(test_full)
    n_val = max(1, int(0.3 * n_test))
    n_train_complement = n_test - n_val
    test_train_sub, val_sub = random_split(
        test_full, [n_train_complement, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = ConcatDataset([good_train, test_train_sub])
    val_ds = val_sub

    # 2. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 3. Modèle, loss, optim, scheduler
    model = build_model(config.model_name, num_classes=2, pretrained=True).to(device)
    # --- Fine-tuning progressif : gel du backbone ---
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    unfreeze_epoch = 5
    logger.info(f"Backbone gelé. Dégel prévu à l'epoch {unfreeze_epoch}.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

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
            "num_train": len(train_loader.dataset),
            "num_val": len(val_loader.dataset),
        })

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(config.epochs):
            if epoch == unfreeze_epoch:
                for param in model.parameters():
                    param.requires_grad = True
                for g in optimizer.param_groups:
                    g["lr"] = config.lr / 10
                logger.info(f"Epoch {epoch+1} : dégel de toutes les couches, LR divisé par 10.")
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

            # --- métriques ---
            metrics = compute_classification_metrics(all_labels, all_preds)
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
        import json
        with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "best_val_loss": best_val_loss,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"],
                "history": history,
            }, f, indent=2)
        mlflow.log_artifact(str(output_dir / "train_metrics.json"))
        mlflow.log_artifact(str(output_dir / "model.pt"))

    return {"best_val_loss": best_val_loss, "final_metrics": metrics}