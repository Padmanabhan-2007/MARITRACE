"""MARITRACE Satellite Model Training and Evaluation Pipeline.

Trains MobileNetV2 with transfer learning, class weighting, LR scheduling,
and early stopping on CSIRO Sentinel-1 SAR imagery.
Produces saved checkpoints, training curves, confusion matrices, and ROC curves.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .satellite_classifier import (
    MobileNetV2SAR,
    SARImageDataset,
    get_data_transforms,
    prepare_stratified_splits,
)


def train_satellite_classifier(
    data_dir: str = r"D:\dataset maritrace\kaggle\data",
    output_dir: str = r"D:\dataset maritrace\outputs\models\satellite_classifier",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 3,
    image_size: int = 224,
    seed: int = 42,
    max_train_samples: int = 0,  # 0 means use all available train samples
) -> Dict[str, Any]:
    """Train and evaluate MobileNetV2 on SAR images with full metrics and artifacts."""
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Satellite Classifier] Training Device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(min(16, os.cpu_count() or 4))
        print(f"  PyTorch CPU multi-threading enabled with {torch.get_num_threads()} threads.")

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Prepare Stratified Splits
    splits = prepare_stratified_splits(data_dir, output_dir, seed=seed)

    train_paths = splits["train"]["paths"]
    train_labels = splits["train"]["labels"]
    val_paths = splits["val"]["paths"]
    val_labels = splits["val"]["labels"]
    test_paths = splits["test"]["paths"]
    test_labels = splits["test"]["labels"]

    if max_train_samples > 0 and max_train_samples < len(train_paths):
        train_paths = train_paths[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
        print(f"  Subsampled training set to {len(train_paths)} for accelerated convergence.")

    # 2. Data Transforms and Loaders
    train_tf, eval_tf = get_data_transforms(image_size=image_size)

    train_ds = SARImageDataset(train_paths, train_labels, transform=train_tf)
    val_ds = SARImageDataset(val_paths, val_labels, transform=eval_tf)
    test_ds = SARImageDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Class Imbalance Weighting
    n_neg = train_labels.count(0)
    n_pos = train_labels.count(1)
    pos_weight = float(n_neg) / max(1, float(n_pos))
    print(f"  Class Balance: Neg(0)={n_neg}, Pos(1)={n_pos} -> pos_weight={pos_weight:.3f}")

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 4. Model, Optimizer & Scheduler
    model = MobileNetV2SAR(pretrained=True, dropout_rate=0.3).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 5. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    best_val_f1 = -1.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    epochs_no_improve = 0

    print(f"\nBeginning training for {epochs} epochs (Early stopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batches = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        scheduler.step()
        train_epoch_loss = running_loss / max(1, batches)

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for val_x, val_y, _ in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device).unsqueeze(1)
                logits = model(val_x)
                loss = criterion(logits, val_y)
                val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_list.extend((probs >= 0.5).astype(int))
                val_targets_list.extend(val_y.cpu().numpy().flatten().astype(int))

        val_epoch_loss = val_loss / max(1, val_batches)
        val_acc = accuracy_score(val_targets_list, val_preds_list)
        val_f1 = f1_score(val_targets_list, val_preds_list, zero_division=0)

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["val_f1"].append(float(val_f1))
        history["val_acc"].append(float(val_acc))

        epoch_elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{epochs}] ({epoch_elapsed:.1f}s) - "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_epoch_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Checkpoint Saving & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "architecture": "MobileNetV2",
                        "image_size": image_size,
                        "pos_weight": pos_weight,
                    },
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint to: {best_model_path} (Val F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without validation F1 improvement.")
                break

    # 6. Evaluation on Unseen Test Set
    print("\n" + "=" * 50)
    print("Evaluating Best Model on Stratified Test Set...")
    print("=" * 50)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_probs_list = []
    test_targets_list = []

    with torch.no_grad():
        for test_x, test_y, _ in test_loader:
            test_x = test_x.to(device)
            logits = model(test_x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            test_probs_list.extend(probs)
            test_targets_list.extend(test_y.numpy().flatten().astype(int))

    test_probs = np.array(test_probs_list)
    test_targets = np.array(test_targets_list)
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds, zero_division=0)
    rec = recall_score(test_targets, test_preds, zero_division=0)
    f1_val = f1_score(test_targets, test_preds, zero_division=0)
    f1_macro = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
    auc = roc_auc_score(test_targets, test_probs)
    cm = confusion_matrix(test_targets, test_preds).tolist()
    clf_report = classification_report(
        test_targets, test_preds, target_names=["No Oil (Class 0)", "Oil (Class 1)"], output_dict=True
    )

    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1 (Oil):  {f1_val:.4f}")
    print(f"Test Macro F1:  {f1_macro:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 7. Generate & Save Diagnostic Plots
    # Plot 1: Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["val_f1"], "g-o", label="Val F1-Score")
    ax2.plot(epochs_range, history["val_acc"], "m--^", label="Val Accuracy")
    ax2.set_title("Validation F1-Score and Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(plots_dir, "satellite_training_curves.png")
    fig.savefig(curves_path, dpi=300)
    plt.close(fig)

    # Plot 2: Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Oil (0)", "Oil (1)"],
        yticklabels=["No Oil (0)", "Oil (1)"],
        ax=ax,
    )
    ax.set_title(f"Satellite SAR Confusion Matrix\nAccuracy: {acc:.3f} | F1: {f1_val:.3f}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Ground Truth")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "satellite_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(test_targets, test_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Sentinel-1 SAR Oil Classifier ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, "satellite_roc_curve.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)

    # 8. Save Metrics & Metadata
    total_training_time = round(time.time() - start_time, 2)
    metrics_summary = {
        "model_architecture": "MobileNetV2 Transfer Learning",
        "total_training_time_seconds": total_training_time,
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": checkpoint["epoch"],
        "device_used": str(device),
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score_oil": round(float(f1_val), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
        },
        "classification_report": clf_report,
        "history": history,
        "artifacts": {
            "checkpoint_path": best_model_path,
            "training_curves_plot": curves_path,
            "confusion_matrix_plot": cm_path,
            "roc_curve_plot": roc_path,
        },
    }

    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nAll satellite model artifacts saved successfully to: {output_dir}")
    return metrics_summary


if __name__ == "__main__":
    train_satellite_classifier(
        data_dir=r"D:\dataset maritrace\kaggle\data",
        output_dir=r"D:\dataset maritrace\outputs\models\satellite_classifier",
        epochs=3,
        batch_size=16,
        lr=3e-4,
        patience=2,
    )
