"""Read-only, memory-aware data access for the MARITRACE dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_project_paths() -> dict[str, str]:
    config_path = ROOT / "maritrace_config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return config.get("paths", {}) if isinstance(config, dict) else {}


_CONFIG_PATHS = _read_project_paths()
_configured_data_root = _CONFIG_PATHS.get("dataset_root")
_configured_output_root = _CONFIG_PATHS.get("primary_outputs_dir")
DATA_ROOTS = [
    Path(os.environ["MARITRACE_DATA_ROOT"]) if os.environ.get("MARITRACE_DATA_ROOT") else None,
    Path(_configured_data_root) if _configured_data_root else None,
    ROOT / "data",
]
OUTPUT_ROOTS = [
    Path(os.environ["MARITRACE_OUTPUT_ROOT"]) if os.environ.get("MARITRACE_OUTPUT_ROOT") else None,
    Path(_configured_output_root) if _configured_output_root else None,
    ROOT / "runs" / "prod_outputs",
    ROOT / "outputs",
]
DATA_ROOTS = [path for path in DATA_ROOTS if path is not None]
OUTPUT_ROOTS = [path for path in OUTPUT_ROOTS if path is not None]


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def output_file(relative: str) -> Path | None:
    return first_existing(*(p / relative for p in OUTPUT_ROOTS))


def data_file(relative: str) -> Path | None:
    return first_existing(*(p / relative for p in DATA_ROOTS))


def load_json(path: Path | None) -> dict[str, Any] | list[Any] | None:
    if not path or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, UnicodeError):
        return None


def load_csv(path: Path | None, nrows: int | None = None) -> pd.DataFrame:
    if not path or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=True)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        return pd.DataFrame()


def inventory() -> list[dict[str, Any]]:
    candidates = {
        "NOAA AIS compressed snapshot": data_file("ais-2024-01-02.csv.zst"),
        "NOAA incidents": data_file("incidents.csv"),
        "Ocean current NetCDF": data_file("cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m_1792237857475.nc"),
        "Reconstructed trajectories": output_file("processed_data/vessel_trajectories.csv"),
        "Filtered AIS positions": output_file("processed_data/ais_filtered_gom.csv"),
        "Backward drift trajectory": output_file("processed_data/drift_backward_trajectory.csv"),
        "Drift origin envelopes": output_file("processed_data/drift_origin_envelopes.geojson"),
        "Audit log": output_file("audit/audit_log.json"),
        "Dataset inventory": output_file("dataset_inventory.csv"),
        "Satellite classifier": output_file("models/satellite_classifier/best_model.pth"),
        "Satellite metrics": output_file("models/satellite_classifier/metrics.json"),
        "Vessel gear classifier": output_file("models/vessel_classifier/vessel_gear_model.joblib"),
        "Vessel metrics": output_file("models/vessel_classifier/metrics.json"),
    }
    rows = []
    for label, path in candidates.items():
        size = path.stat().st_size if path else None
        status = "Missing" if path is None else "Invalid · 0 bytes" if size == 0 else "Available"
        rows.append({"Artifact": label, "Status": status, "Path": str(path) if path else "—", "Size": _size(path) if path else "—"})
    for scene in sorted((ROOT / "data" / "sample").glob("scene-*")):
        for rel, label in [("ais.csv", "AIS sample"), ("polygons.geojson", "Slick polygon sample"), ("metadata.json", "Scene metadata")]:
            path = scene / rel
            rows.append({"Artifact": f"{scene.name} {label}", "Status": "Available" if path.is_file() else "Missing", "Path": str(path) if path.is_file() else "—", "Size": _size(path) if path.is_file() else "—"})
    for filename in ["drifting_longlines.csv", "fixed_gear.csv", "pole_and_line.csv", "purse_seines.csv", "trawlers.csv", "trollers.csv", "unknown.csv"]:
        path = data_file(filename)
        rows.append({"Artifact": f"GFW source · {filename}", "Status": "Available" if path else "Missing", "Path": str(path) if path else "—", "Size": _size(path) if path else "—"})
    return rows


def _size(path: Path) -> str:
    size = path.stat().st_size
    return f"{size / (1024**3):.2f} GB" if size >= 1024**3 else f"{size / (1024**2):.1f} MB" if size >= 1024**2 else f"{size / 1024:.1f} KB"

