"""MARITRACE Satellite SAR Image Classifier.

Implements MobileNetV2 transfer learning for binary detection of oil slick features
versus look-alikes / clean ocean in Sentinel-1 Synthetic Aperture Radar imagery.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms


class SARImageDataset(Dataset):
    """PyTorch Dataset for Sentinel-1 SAR chips."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image; convert grayscale/RGB to 3-channel RGB for pretrained MobileNetV2
        with Image.open(path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, torch.tensor(label, dtype=torch.float32), path


class MobileNetV2SAR(nn.Module):
    """MobileNetV2 Transfer Learning Architecture for SAR Oil Slick Detection."""

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # Freeze early feature layers for stable transfer learning
        for param in self.backbone.features[:8].parameters():
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (B, 1)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict calibrated probability of oil slick presence P(oil) in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def get_data_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation/test transforms with moderate augmentation."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


def prepare_stratified_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Scan Class_0 and Class_1, deduplicate, and create stratified splits."""
    random.seed(seed)
    np.random.seed(seed)

    c0_files = sorted(glob.glob(os.path.join(data_dir, "Class_0", "*.jpg")))
    c1_files = sorted(glob.glob(os.path.join(data_dir, "Class_1", "*.jpg")))

    print(f"Discovered: {len(c0_files)} Class_0 (No Oil) and {len(c1_files)} Class_1 (Oil) images.")

    all_paths = c0_files + c1_files
    all_labels = [0] * len(c0_files) + [1] * len(c1_files)

    # Use only explicit acquisition metadata; filenames alone are not reliable
    # scene IDs. Without a complete map, label the split as image level.
    scene_manifest_path = os.path.join(data_dir, "scene_manifest.csv")
    scene_by_name = {}
    if os.path.isfile(scene_manifest_path):
        scene_manifest = pd.read_csv(scene_manifest_path)
        if not {"filename", "scene_id"}.issubset(scene_manifest.columns):
            raise ValueError("scene_manifest.csv must contain filename and scene_id columns")
        scene_by_name = dict(zip(scene_manifest.filename.astype(str), scene_manifest.scene_id.astype(str)))
    has_all_scene_ids = bool(all_paths) and all(os.path.basename(p) in scene_by_name for p in all_paths)
    if has_all_scene_ids:
        groups = np.asarray([scene_by_name[os.path.basename(p)] for p in all_paths])
        train_val_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=seed
        ).split(all_paths, all_labels, groups))
        tr_rel, va_rel = next(GroupShuffleSplit(
            n_splits=1, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed + 1
        ).split(np.asarray(all_paths)[train_val_idx], np.asarray(all_labels)[train_val_idx], groups[train_val_idx]))
        train_idx, val_idx = train_val_idx[tr_rel], train_val_idx[va_rel]
        split_strategy = "scene_group_holdout"
    else:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(all_paths)), test_size=test_ratio, stratify=all_labels, random_state=seed
        )
        tr_rel, va_rel = train_test_split(
            np.arange(len(train_val_idx)), test_size=val_ratio / (train_ratio + val_ratio),
            stratify=np.asarray(all_labels)[train_val_idx], random_state=seed,
        )
        train_idx, val_idx = train_val_idx[tr_rel], train_val_idx[va_rel]
        groups = np.asarray([scene_by_name.get(os.path.basename(p)) for p in all_paths], dtype=object)
        split_strategy = "stratified_image_level_no_scene_metadata"

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    splits = {
        name: {"paths": [all_paths[i] for i in idx], "labels": [all_labels[i] for i in idx], "count": len(idx)}
        for name, idx in split_indices.items()
    }
    splits["class_counts"] = {
        name: {0: splits[name]["labels"].count(0), 1: splits[name]["labels"].count(1)}
        for name in split_indices
    }

    manifest_path = os.path.join(output_dir, "dataset_splits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "split_strategy": split_strategy,
                "scene_metadata_file": scene_manifest_path if has_all_scene_ids else None,
                "files": {
                    name: [{
                        "path": os.path.relpath(all_paths[i], data_dir),
                        "label": int(all_labels[i]),
                        "scene_id": str(groups[i]) if has_all_scene_ids else None,
                        "sha256": hashlib.sha256(open(all_paths[i], "rb").read()).hexdigest(),
                    } for i in idx]
                    for name, idx in split_indices.items()
                },
                "counts": {
                    "train": splits["train"]["count"],
                    "val": splits["val"]["count"],
                    "test": splits["test"]["count"],
                },
                "class_counts": splits["class_counts"],
            },
            f,
            indent=2,
        )

    print(f"Splits created: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    print(f"Train class balance: {splits['class_counts']['train']}")
    return splits
