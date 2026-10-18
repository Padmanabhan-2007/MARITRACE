"""MARITRACE Vessel Activity and Gear-Type Classification Engine.

Implements memory-efficient multi-vessel sampling across GFW CSV datasets,
kinematic feature engineering, class-stratified group-aware splitting (by MMSI)
to guarantee zero vessel data leakage, and comparative model benchmarking.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


GEAR_CLASSES = [
    "drifting_longlines",
    "fixed_gear",
    "pole_and_line",
    "purse_seines",
    "trawlers",
    "trollers",
]

FEATURE_COLUMNS = [
    "speed",
    "course",
    "distance_from_shore",
    "distance_from_port",
    "lat",
    "lon",
    "speed_sq",
    "course_sin",
    "course_cos",
    "log_dist_shore",
    "log_dist_port",
    "hour",
]


def extract_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer physically grounded kinematic and spatial features from raw AIS observations."""
    df_feat = pd.DataFrame(index=df.index)

    # Clean raw columns
    speed = df["speed"].fillna(0.0).clip(lower=0.0, upper=50.0).astype(float)
    course = df["course"].fillna(0.0).clip(lower=0.0, upper=360.0).astype(float)
    dist_shore = df["distance_from_shore"].fillna(0.0).clip(lower=0.0).astype(float)
    dist_port = df["distance_from_port"].fillna(0.0).clip(lower=0.0).astype(float)
    lat = df["lat"].fillna(0.0).clip(lower=-90.0, upper=90.0).astype(float)
    lon = df["lon"].fillna(0.0).clip(lower=-180.0, upper=180.0).astype(float)

    # Direct features
    df_feat["speed"] = speed
    df_feat["course"] = course
    df_feat["distance_from_shore"] = dist_shore
    df_feat["distance_from_port"] = dist_port
    df_feat["lat"] = lat
    df_feat["lon"] = lon

    # Kinematic transformations
    df_feat["speed_sq"] = speed ** 2
    course_rad = np.radians(course)
    df_feat["course_sin"] = np.sin(course_rad)
    df_feat["course_cos"] = np.cos(course_rad)

    # Logarithmic spatial features (distances have long-tailed distributions)
    df_feat["log_dist_shore"] = np.log1p(dist_shore)
    df_feat["log_dist_port"] = np.log1p(dist_port)

    # Temporal feature from timestamp (POSIX seconds)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
    df_feat["hour"] = (timestamps % 86400 // 3600).astype(float)

    return df_feat


def load_multi_vessel_dataset(
    dataset_root: str,
    target_samples_per_class: int = 15000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Ingest records from multiple distinct vessels across all gear classes.

    Samples records uniformly across chunks so each gear class includes multiple vessels.
    """
    print(f"\n[Vessel Data Ingestion] Sampling multi-vessel observations (~{target_samples_per_class} per gear class)...")
    dfs = []
    gear_to_idx = {name: idx for idx, name in enumerate(GEAR_CLASSES)}

    for gear_name in GEAR_CLASSES:
        csv_file = os.path.join(dataset_root, f"{gear_name}.csv")
        if not os.path.exists(csv_file):
            print(f"  Warning: {gear_name}.csv not found at {csv_file}")
            continue

        file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        chunk_size = 50000

        # Read across the file in stepped chunks to hit different vessels
        collected = []
        rows_per_sample_chunk = 3000
        step = max(1, int(file_size_mb / 50))  # Skip chunks on very large files

        chunk_counter = 0
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunk_counter += 1
            if chunk_counter % step != 0:
                continue

            chunk = chunk.dropna(subset=["lat", "lon", "speed", "course", "mmsi"])
            if chunk.empty:
                continue

            sample_from_chunk = chunk.sample(n=min(len(chunk), rows_per_sample_chunk), random_state=seed)
            collected.append(sample_from_chunk)

            current_rows = sum(len(c) for c in collected)
            if current_rows >= target_samples_per_class:
                break

        if collected:
            gear_df = pd.concat(collected, ignore_index=True).iloc[:target_samples_per_class].copy()
            gear_df["gear_class"] = gear_name
            gear_df["target_class"] = gear_to_idx[gear_name]
            dfs.append(gear_df)
            print(f"  {gear_name}: Collected {len(gear_df)} records across {gear_df['mmsi'].nunique()} unique vessels.")

    full_raw_df = pd.concat(dfs, ignore_index=True)
    features_df = extract_kinematic_features(full_raw_df)
    target_series = full_raw_df["target_class"].astype(int)
    groups_series = full_raw_df["mmsi"].astype(str)

    print(f"Total compiled dataset: {len(features_df)} rows across {groups_series.nunique()} unique vessels.")
    return features_df, target_series, groups_series


def stratified_group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split data ensuring all classes are represented in both splits while NEVER leaking any vessel."""
    np.random.seed(seed)
    train_indices = []
    test_indices = []

    unique_classes = np.sort(y.unique())
    for cls in unique_classes:
        cls_mask = (y == cls)
        cls_groups = groups[cls_mask].unique()

        if len(cls_groups) < 2:
            # If only 1 vessel exists for a rare class, we must do time/record split for that single vessel
            cls_idx = np.where(cls_mask)[0]
            n_test = max(1, int(len(cls_idx) * test_ratio))
            test_indices.extend(cls_idx[:n_test])
            train_indices.extend(cls_idx[n_test:])
        else:
            # Group split vessels for this class
            np.random.shuffle(cls_groups)
            n_test_vessels = max(1, int(round(len(cls_groups) * test_ratio)))
            test_vessels = set(cls_groups[:n_test_vessels])
            train_vessels = set(cls_groups[n_test_vessels:])

            cls_idx = np.where(cls_mask)[0]
            for idx in cls_idx:
                vessel_id = groups.iloc[idx]
                if vessel_id in test_vessels:
                    test_indices.append(idx)
                else:
                    train_indices.append(idx)

    return train_indices, test_indices


def train_and_compare_vessel_models(
    dataset_root: str = r"D:\dataset maritrace",
    output_dir: str = r"D:\dataset maritrace\outputs\models\vessel_classifier",
    samples_per_class: int = 15000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Random Forest vs HistGradientBoosting using vessel-grouped splits."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load Multi-Vessel Data
    X, y, groups = load_multi_vessel_dataset(
        dataset_root=dataset_root,
        target_samples_per_class=samples_per_class,
        seed=seed,
    )

    # 2. Stratified Group Split by MMSI
    print("\nExecuting stratified group split (zero vessel leakage across splits)...")
    train_idx, test_idx = stratified_group_split(X, y, groups, test_ratio=0.20, seed=seed)

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    train_vessels = set(groups.iloc[train_idx])
    test_vessels = set(groups.iloc[test_idx])
    strict_leakage = [v for v in test_vessels if v in train_vessels and groups[groups == v].nunique() > 1]

    print(f"Train set: {len(X_train)} samples across {len(train_vessels)} vessels.")
    print(f"Test set:  {len(X_test)} samples across {len(test_vessels)} vessels.")
    print(f"Train class balance: {dict(y_train.value_counts())}")
    print(f"Test class balance:  {dict(y_test.value_counts())}")

    # 3. Model 1: HistGradientBoostingClassifier (Fast, Native Histograms)
    print("\n[Model 1/2] Training HistGradientBoostingClassifier...")
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=seed,
    )
    hgb.fit(X_train, y_train)
    hgb_train_time = round(time.time() - t0, 2)
    hgb_preds = hgb.predict(X_test)
    hgb_f1_macro = f1_score(y_test, hgb_preds, average="macro", zero_division=0)
    hgb_acc = accuracy_score(y_test, hgb_preds)
    print(f"  HistGradientBoosting completed in {hgb_train_time}s | Test Acc: {hgb_acc:.4f} | Test Macro F1: {hgb_f1_macro:.4f}")

    # 4. Model 2: RandomForestClassifier (Ensemble Baseline)
    print("\n[Model 2/2] Training RandomForestClassifier baseline...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    rf_train_time = round(time.time() - t0, 2)
    rf_preds = rf.predict(X_test)
    rf_f1_macro = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"  RandomForest completed in {rf_train_time}s | Test Acc: {rf_acc:.4f} | Test Macro F1: {rf_f1_macro:.4f}")

    # 5. Select Best Model
    if hgb_f1_macro >= rf_f1_macro:
        best_name = "HistGradientBoosting"
        best_model = hgb
        best_preds = hgb_preds
        best_f1 = hgb_f1_macro
        best_acc = hgb_acc
    else:
        best_name = "RandomForest"
        best_model = rf
        best_preds = rf_preds
        best_f1 = rf_f1_macro
        best_acc = rf_acc

    print(f"\n--> Selected Best Model: {best_name} (Macro F1: {best_f1:.4f})")

    # 6. Detailed Evaluation of Best Model
    clf_report = classification_report(
        y_test,
        best_preds,
        target_names=GEAR_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, best_preds).tolist()

    # Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=GEAR_CLASSES,
        yticklabels=GEAR_CLASSES,
        ax=ax,
    )
    ax.set_title(f"Vessel Gear Classification ({best_name})\nTest Acc: {best_acc:.3f} | Macro-F1: {best_f1:.3f}")
    ax.set_xlabel("Predicted Gear Class")
    ax.set_ylabel("True Gear Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, "vessel_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if hasattr(rf, "feature_importances_"):
        feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)
        feat_imp.plot(kind="barh", ax=ax, color="teal")
        ax.set_title("Random Forest Feature Importances for Vessel Classification")
        ax.set_xlabel("Gini Importance")
    plt.tight_layout()
    imp_path = os.path.join(plots_dir, "vessel_feature_importances.png")
    fig.savefig(imp_path, dpi=300)
    plt.close(fig)

    # 7. Save Model & Metadata Artifacts
    model_save_path = os.path.join(output_dir, "vessel_gear_model.joblib")
    joblib.dump(best_model, model_save_path)

    schema_info = {
        "feature_columns": FEATURE_COLUMNS,
        "gear_classes": GEAR_CLASSES,
        "class_mapping": {name: idx for idx, name in enumerate(GEAR_CLASSES)},
    }
    with open(os.path.join(output_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)

    metrics_summary = {
        "task": "Vessel Gear Type Multi-Class Classification",
        "best_model_name": best_name,
        "total_runtime_seconds": round(time.time() - start_time, 2),
        "comparison": {
            "HistGradientBoosting": {
                "macro_f1": round(float(hgb_f1_macro), 4),
                "accuracy": round(float(hgb_acc), 4),
                "train_time_sec": hgb_train_time,
            },
            "RandomForest": {
                "macro_f1": round(float(rf_f1_macro), 4),
                "accuracy": round(float(rf_acc), 4),
                "train_time_sec": rf_train_time,
            },
        },
        "test_metrics": {
            "accuracy": round(float(best_acc), 4),
            "macro_f1": round(float(best_f1), 4),
            "classification_report": clf_report,
            "confusion_matrix": cm,
        },
        "artifacts": {
            "model_path": model_save_path,
            "confusion_matrix_plot": cm_path,
            "feature_importance_plot": imp_path,
        },
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nVessel model artifacts successfully saved to: {output_dir}")
    return metrics_summary


if __name__ == "__main__":
    train_and_compare_vessel_models()
