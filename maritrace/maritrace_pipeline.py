"""MARITRACE Master Pipeline Runner and CLI Interface.

Orchestrates dataset discovery, satellite SAR image model training, vessel model training,
AIS streaming trajectory extraction, ocean drift simulation, multi-source evidence fusion,
evidence sufficiency assessment, and cryptographic audit logging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from src.ais_engine import reconstruct_vessel_trajectories, stream_filter_ais_zst
from src.audit_logger import AuditTrail
from src.discovery import run_full_discovery
from src.evidence_fusion import EvidenceFusionEngine
from src.llm_advisor import generate_forensic_briefing
from src.ocean_drift import run_drift_pipeline
from src.satellite_classifier import MobileNetV2SAR, get_data_transforms
from src.sufficiency_engine import NextBestEvidenceEngine
from src.train_satellite import train_satellite_classifier
from src.vessel_classifier import train_and_compare_vessel_models


def load_config(config_path: str = "maritrace_config.yaml") -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    stage: str = "all",
    force_retrain: bool = False,
    config_path: str = "maritrace_config.yaml",
) -> Dict[str, Any]:
    """Execute MARITRACE pipeline stages with restartability and audit provenance."""
    t_start = time.time()
    cfg = load_config(config_path)

    dataset_root = cfg["paths"]["dataset_root"]
    output_root = cfg["paths"]["primary_outputs_dir"]
    os.makedirs(output_root, exist_ok=True)

    audit = AuditTrail(log_dir=os.path.join(output_root, "audit"))
    audit.log_event("PIPELINE_INVOCATION", {
        "stage": stage,
        "force_retrain": force_retrain,
        "config": cfg["project"],
    })

    print("=" * 75)
    print("MARITRACE — MARITIME POLLUTION FORENSICS & EVIDENCE FUSION SYSTEM")
    print(f"Version: {cfg['project']['version']} | Datathon Mode: Active | Hardware Cores: {os.cpu_count()}")
    print(f"Target Output Directory: {output_root}")
    print("=" * 75)

    pipeline_results: Dict[str, Any] = {}

    # STAGE 1: DATASET DISCOVERY
    if stage in ["all", "inventory"]:
        print("\n>>> STAGE 1: DATASET DISCOVERY & AUDIT")
        inv_csv = os.path.join(output_root, "dataset_inventory.csv")
        if os.path.exists(inv_csv) and not force_retrain:
            print(f"  Existing dataset inventory found at {inv_csv}, loading...")
            df_inv = pd.read_csv(inv_csv)
        else:
            df_inv, audit_summary = run_full_discovery(dataset_root, output_root)
        pipeline_results["inventory_count"] = len(df_inv)
        audit.log_event("STAGE_INVENTORY_COMPLETED", {"datasets_cataloged": len(df_inv)})

    # STAGE 2: SATELLITE SAR MODEL TRAINING
    if stage in ["all", "satellite"]:
        print("\n>>> STAGE 2: SATELLITE SAR CLASSIFIER")
        model_dir = os.path.join(output_root, "models", "satellite_classifier")
        best_ckpt = os.path.join(model_dir, "best_model.pth")
        if os.path.exists(best_ckpt) and not force_retrain:
            print(f"  Existing satellite model checkpoint found at {best_ckpt}. Skipping retraining.")
            with open(os.path.join(model_dir, "metrics.json"), "r", encoding="utf-8") as f:
                sat_metrics = json.load(f)
        else:
            sat_cfg = cfg.get("satellite_model", {})
            sat_metrics = train_satellite_classifier(
                data_dir=cfg["paths"]["sar_images_dir"],
                output_dir=model_dir,
                epochs=sat_cfg.get("epochs", 3),
                batch_size=sat_cfg.get("batch_size", 16),
                lr=sat_cfg.get("learning_rate", 3e-4),
                patience=sat_cfg.get("early_stopping_patience", 2),
            )
        pipeline_results["satellite_metrics"] = sat_metrics.get("test_metrics")
        audit.log_event("STAGE_SATELLITE_COMPLETED", {"test_metrics": sat_metrics.get("test_metrics")})

    # STAGE 3: VESSEL MODEL TRAINING
    if stage in ["all", "vessel"]:
        print("\n>>> STAGE 3: VESSEL GEAR & ACTIVITY MODEL")
        vessel_model_dir = os.path.join(output_root, "models", "vessel_classifier")
        best_vessel_model = os.path.join(vessel_model_dir, "vessel_gear_model.joblib")
        if os.path.exists(best_vessel_model) and not force_retrain:
            print(f"  Existing vessel model found at {best_vessel_model}. Skipping retraining.")
            with open(os.path.join(vessel_model_dir, "metrics.json"), "r", encoding="utf-8") as f:
                vessel_metrics = json.load(f)
        else:
            vessel_metrics = train_and_compare_vessel_models(
                dataset_root=dataset_root,
                output_dir=vessel_model_dir,
                samples_per_class=15000,
            )
        pipeline_results["vessel_metrics"] = vessel_metrics.get("test_metrics")
        audit.log_event("STAGE_VESSEL_COMPLETED", {"best_model": vessel_metrics.get("best_model_name")})

    # STAGE 4: AIS STREAMING & TRAJECTORY RECONSTRUCTION
    if stage in ["all", "ais"]:
        print("\n>>> STAGE 4: AIS STREAMING & TRAJECTORY FORENSICS")
        zst_file = os.path.join(dataset_root, cfg["ais"]["file_name"])
        out_filtered_ais = os.path.join(output_root, "processed_data", "ais_filtered_gom.csv")
        out_trajectories = os.path.join(output_root, "processed_data", "vessel_trajectories.csv")

        if os.path.exists(out_trajectories) and not force_retrain:
            print(f"  Existing reconstructed trajectories found at {out_trajectories}.")
            trajectories_df = pd.read_csv(out_trajectories)
        else:
            roi = cfg["ais"]["default_roi"]
            stream_filter_ais_zst(
                zst_path=zst_file,
                output_csv_path=out_filtered_ais,
                roi_lat=(roi["lat_min"], roi["lat_max"]),
                roi_lon=(roi["lon_min"], roi["lon_max"]),
                max_records=cfg["ais"].get("stream_sample_limit", 1500000),
            )
            inc = cfg["incident"]
            trajectories_df = reconstruct_vessel_trajectories(
                filtered_csv_path=out_filtered_ais,
                incident_lat=inc["lat"],
                incident_lon=inc["lon"],
                incident_time=inc.get("observation_time_utc", "2024-01-02 12:00:00"),
                output_summary_csv=out_trajectories,
            )
        pipeline_results["trajectories_count"] = len(trajectories_df)
        audit.log_event("STAGE_AIS_COMPLETED", {"distinct_vessels": len(trajectories_df)})

    # STAGE 5: OCEAN CURRENT DRIFT FORENSICS
    if stage in ["all", "drift"]:
        print("\n>>> STAGE 5: OCEAN CURRENT VALIDATION & DRIFT HINDCAST")
        nc_file = os.path.join(dataset_root, cfg["ocean_drift"]["netcdf_file"])
        inc = cfg["incident"]
        drift_res = run_drift_pipeline(
            netcdf_path=nc_file,
            output_dir=os.path.join(output_root, "processed_data"),
            plots_dir=os.path.join(output_root, "plots"),
            spill_lat=inc["lat"],
            spill_lon=inc["lon"],
        )
        pipeline_results["drift_metadata"] = drift_res
        audit.log_event("STAGE_DRIFT_COMPLETED", {"provenance": drift_res["provenance_mode"]})

    # STAGE 6 & 7: EVIDENCE FUSION & NEXT-BEST-EVIDENCE RECOMMENDATION
    if stage in ["all", "fusion"]:
        print("\n>>> STAGE 6 & 7: EVIDENCE FUSION & SUFFICIENCY ENGINE")
        inc = cfg["incident"]
        out_trajectories = os.path.join(output_root, "processed_data", "vessel_trajectories.csv")
        trajectories_df = pd.read_csv(out_trajectories) if os.path.exists(out_trajectories) else pd.DataFrame()

        drift_csv = os.path.join(output_root, "processed_data", "drift_backward_trajectory.csv")
        drift_steps = pd.read_csv(drift_csv).to_dict(orient="records") if os.path.exists(drift_csv) else []

        # No incident-linked SAR scene is configured. Held-out test accuracy is
        # not an incident-specific detection score, so fusion must abstain here.
        sat_oil_confidence = None

        fusion_engine = EvidenceFusionEngine(
            weights=cfg.get("fusion", {}).get("weights"),
            audit_trail=audit,
        )
        fusion_result = fusion_engine.fuse_incident_evidence(
            incident_id=str(inc.get("default_incident_id", "10716")),
            incident_name=inc.get("name", "Mystery Sheen; LA"),
            spill_lat=inc["lat"],
            spill_lon=inc["lon"],
            sar_oil_prob=sat_oil_confidence,
            drift_steps=drift_steps,
            vessel_trajectories=trajectories_df,
        )

        sufficiency_engine = NextBestEvidenceEngine(policy="balanced")
        sufficiency_result = sufficiency_engine.evaluate_incident_sufficiency(fusion_result)

        # Generate briefing
        briefing = generate_forensic_briefing(
            fusion_result,
            sufficiency_result,
            drift_metadata=pipeline_results.get("drift_metadata"),
        )
        briefing_path = os.path.join(output_root, "FORENSIC_INCIDENT_DOSSIER.md")
        with open(briefing_path, "w", encoding="utf-8") as f:
            f.write(briefing)

        pipeline_results["fusion_result"] = fusion_result
        pipeline_results["sufficiency_result"] = sufficiency_result
        print(f"\nSaved Comprehensive Forensic Dossier to: {briefing_path}")

        print("\n" + "=" * 65)
        print("EXECUTIVE FORENSIC SUMMARY")
        print("=" * 65)
        print(f"Incident: {fusion_result['incident_name']} ({inc['lat']}, {inc['lon']})")
        print(f"Decision: {fusion_result['decision']}")
        print(f"Reason:   {fusion_result['primary_reason']}")
        print(f"Top Lead: {fusion_result.get('top_candidate', 'None (Abstained)')}")
        print(f"Next Best Evidence to Acquire:")
        if sufficiency_result.get("primary_recommendation"):
            rec = sufficiency_result["primary_recommendation"]
            print(f"  -> {rec['name']} (Priority Score: {rec['priority_score']}, Cost: ${rec['cost_usd']})")
        print("=" * 65)

    # Integrity verification
    valid_chain, error = audit.verify_integrity()
    audit.log_event("PIPELINE_EXECUTION_FINISHED", {
        "duration_seconds": round(time.time() - t_start, 2),
        "audit_integrity_valid": valid_chain,
    })
    print(f"\nAudit Log: {len(audit.chain)} blocks written. Cryptographic integrity verified: {valid_chain}")

    # Save summary
    summary_path = os.path.join(output_root, "processed_data", "maritrace_execution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runtime_seconds": round(time.time() - t_start, 2),
            "audit_chain_blocks": len(audit.chain),
            "audit_integrity_verified": valid_chain,
            "results": pipeline_results,
        }, f, indent=2, default=str)

    print(f"MARITRACE Pipeline finished in {time.time() - t_start:.2f}s.")
    return pipeline_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARITRACE Maritime Pollution Forensics Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "inventory", "satellite", "vessel", "ais", "drift", "fusion"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution/retraining even if cached checkpoints exist",
    )
    parser.add_argument(
        "--config",
        default="maritrace_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    run_pipeline(stage=args.stage, force_retrain=args.force, config_path=args.config)
