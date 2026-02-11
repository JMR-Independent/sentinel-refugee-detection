"""Training loop and evaluation utilities."""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_dataset, val_dataset, config, device="cpu",
                checkpoint_dir="checkpoints"):
    """Train the camp classifier with early stopping.

    Parameters
    ----------
    model : nn.Module
        The classifier model.
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset
        Validation dataset.
    config : dict
        Full configuration dict.
    device : str
        'cpu', 'cuda', or 'mps'.
    checkpoint_dir : str
        Directory to save model checkpoints.

    Returns
    -------
    dict
        Training history with losses and metrics per epoch.
    """
    from src.model import freeze_backbone, unfreeze_all

    mc = config["model"]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Loss with class imbalance weighting
    pos_weight = torch.tensor([mc["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=mc["lr"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=mc["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=mc["batch_size"], shuffle=False, num_workers=0,
    )

    # Training state
    history = {"train_loss": [], "val_loss": [], "val_precision": [],
               "val_recall": [], "val_f1": [], "val_auc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    # Freeze backbone for initial epochs
    freeze_backbone(model)

    for epoch in range(mc["max_epochs"]):
        # Unfreeze after freeze_epochs
        if epoch == mc["freeze_epochs"]:
            unfreeze_all(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=mc["lr"] * 0.1)
            print(f"Epoch {epoch}: unfreezing all layers, reducing lr")

        # --- Train ---
        model.train()
        train_losses = []
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        history["train_loss"].append(train_loss)

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, criterion, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{mc['max_epochs']} "
              f"[{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"P={val_metrics['precision']:.3f} "
              f"R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} "
              f"AUC={val_metrics['auc']:.3f}")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(),
                       checkpoint_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= mc["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(
        torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    )
    return history


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset.

    Parameters
    ----------
    model : nn.Module
        The classifier.
    data_loader : DataLoader
        Data to evaluate on.
    criterion : nn.Module
        Loss function.
    device : str
        Device.

    Returns
    -------
    dict
        Metrics: loss, precision, recall, f1, auc.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    model.eval()
    all_labels = []
    all_probs = []
    losses = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    # Handle edge cases where only one class is present
    if len(np.unique(all_labels)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": np.mean(losses),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": auc,
        "labels": all_labels,
        "probs": all_probs,
    }


def predict(model, data_loader, device="cpu"):
    """Run inference and return probabilities.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    data_loader : DataLoader
        Data to predict on.
    device : str
        Device.

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


# ---------------------------------------------------------------------------
# Precision @ Top-K (operationally useful metric)
# ---------------------------------------------------------------------------

def precision_at_top_k(labels, probs, k=50):
    """Compute precision among the top-K most confident predictions.

    This answers: "If we investigate our top K detections, how many are real?"
    More operationally useful than global precision for humanitarian response.

    Parameters
    ----------
    labels : np.ndarray
        True binary labels.
    probs : np.ndarray
        Predicted probabilities.
    k : int
        Number of top predictions to evaluate.

    Returns
    -------
    float
        Precision among top-K predictions.
    dict
        Details with 'k', 'true_positives', 'indices'.
    """
    k = min(k, len(probs))
    top_k_indices = np.argsort(probs)[::-1][:k]
    top_k_labels = labels[top_k_indices]

    tp = int(top_k_labels.sum())
    precision = tp / k if k > 0 else 0.0

    return precision, {
        "k": k,
        "true_positives": tp,
        "false_positives": k - tp,
        "indices": top_k_indices,
        "min_prob": float(probs[top_k_indices[-1]]) if k > 0 else 0.0,
    }
