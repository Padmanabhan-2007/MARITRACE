"""Comprehensive Unit and Integration Test Suite for MARITRACE Pipeline."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.ais_engine import haversine_km, reconstruct_vessel_trajectories
from src.audit_logger import AuditRecord, AuditTrail
from src.evidence_fusion import EvidenceFusionEngine
from src.ocean_drift import OceanDriftEngine
from src.satellite_classifier import MobileNetV2SAR
from src.sufficiency_engine import NextBestEvidenceEngine
from src.vessel_classifier import extract_kinematic_features


def test_haversine_distance():
    """Verify haversine distance calculation."""
    # Distance from (0, 0) to (0, 1) degree longitude at equator is ~111.32 km
    dist = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert 111.0 <= dist <= 112.0

    # Distance to self is 0
    assert haversine_km(29.135, -89.304, 29.135, -89.304) == 0.0


def test_netcdf_validation_corrupt_file():
    """Verify that a 0-byte NetCDF file is explicitly flagged as corrupt."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        temp_path = f.name
    try:
        # File has 0 bytes
        res = OceanDriftEngine.validate_netcdf(temp_path)
        assert res["valid"] is False
        assert res["status"] == "CORRUPTED_ZERO_BYTES"
        assert res["size_bytes"] == 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_satellite_mobilenet_forward():
    """Verify MobileNetV2 SAR model architecture and output shapes."""
    model = MobileNetV2SAR(pretrained=False)
    model.eval()

    # Input tensor batch of 2 RGB images (224 x 224)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    # Probabilities must be strictly bounded in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_vessel_feature_engineering():
    """Verify kinematic and spatial feature engineering."""
    raw_df = pd.DataFrame({
        "speed": [12.5, np.nan, -1.0],
        "course": [180.0, 360.0, np.nan],
        "distance_from_shore": [5000.0, 0.0, np.nan],
        "distance_from_port": [10000.0, 500.0, np.nan],
        "lat": [29.0, 30.0, 28.5],
        "lon": [-89.0, -90.0, -88.5],
        "timestamp": [1704196800, 1704200400, 1704204000],
    })

    feats = extract_kinematic_features(raw_df)
    assert "speed" in feats.columns
    assert "course_sin" in feats.columns
    assert "course_cos" in feats.columns
    assert "log_dist_shore" in feats.columns
    assert feats.isna().sum().sum() == 0  # No NaNs in engineered features


def test_ocean_drift_simulation():
    """Verify Lagrangian back-drift numerical integration and uncertainty radius."""
    engine = OceanDriftEngine(wind_factor=0.03, coriolis_deflection_deg=12.0, eddy_diffusivity_m2s=15.0)
    sim = engine.simulate_back_drift(
        spill_lat=29.1350,
        spill_lon=-89.3039,
        hours_list=[2, 6, 12],
        cur_u=0.18,
        cur_v=-0.08,
        wind_u=-3.5,
        wind_v=-2.0,
    )

    steps = sim["trajectory_steps"]
    assert len(steps) == 3
    # Uncertainty radius must increase monotonically over time
    r2 = next(s["uncertainty_radius_km"] for s in steps if s["hours_back"] == 2.0)
    r12 = next(s["uncertainty_radius_km"] for s in steps if s["hours_back"] == 12.0)
    assert r12 > r2
    assert "features" in sim["geojson"]


def test_evidence_fusion_abstention_on_lookalike(tmp_path):
    """Verify that the evidence fusion engine abstains when SAR detects a lookalike."""
    engine = EvidenceFusionEngine(threshold_sar_min=0.50, audit_trail=AuditTrail(log_dir=str(tmp_path)))
    mock_trajectories = pd.DataFrame([{
        "mmsi": "367747070",
        "vessel_name": "MISS GWEN",
        "vessel_type": 70,
        "min_distance_to_spill_km": 1.2,
        "max_ais_gap_hours": 0.1,
        "loiter_fraction": 0.05,
        "hours_from_spill_window": 0.5,
    }])

    # SAR oil probability 0.25 (Look-alike / wind calm / biogenic)
    res = engine.fuse_incident_evidence(
        incident_id="TEST-01",
        incident_name="Test Lookalike Scene",
        spill_lat=29.135,
        spill_lon=-89.304,
        sar_oil_prob=0.25,
        drift_steps=[{"hours_back": 6.0, "mean_lat": 29.135, "mean_lon": -89.304, "uncertainty_radius_km": 5.0}],
        vessel_trajectories=mock_trajectories,
    )

    assert res["decision"] == "ABSTAIN"
    assert res["status"] == "NOT_AN_OIL_SLICK"
    assert len(res["ranked_leads"]) == 0


def test_historical_ais_tracks_never_enter_incident_rankings(tmp_path):
    engine = EvidenceFusionEngine(audit_trail=AuditTrail(log_dir=str(tmp_path)))
    tracks = pd.DataFrame([{
        "mmsi": "123456789", "vessel_name": "BASELINE ONLY",
        "start_time_utc": "2024-01-02T00:00:00Z", "end_time_utc": "2024-01-02T23:59:00Z",
        "coverage_min_lat": 29.0, "coverage_max_lat": 29.2,
        "coverage_min_lon": -89.5, "coverage_max_lon": -89.2,
    }])
    result = engine.fuse_incident_evidence(
        incident_id="10716", incident_name="Historical alignment check",
        spill_lat=29.135, spill_lon=-89.304, sar_oil_prob=0.9,
        drift_steps=[], vessel_trajectories=tracks, incident_time_utc="2024-01-07T10:00:00Z",
    )
    assert result["decision"] == "ABSTAIN"
    assert result["ranked_leads"] == []
    assert result["candidate_count"] == 0
    assert result["is_historical_baseline"] is True
    assert len(result["historical_baseline_vessels"]) == 1


def test_uncertainty_reduction_relative_math_is_consistent():
    engine = NextBestEvidenceEngine()
    result = engine.evaluate_incident_sufficiency({
        "incident_id": "TEST", "decision": "AMBIGUOUS_LEADS", "ranked_leads": []
    })
    option = next(x for x in result["ranked_acquisition_options"]
                  if x["id"] == "COMMERCIAL_SATELLITE_AIS")
    assert option["prior_uncertainty"] == 0.65
    assert option["projected_posterior_uncertainty"] == 0.30
    assert option["relative_uncertainty_reduction_pct"] == 53.8
    assert "not a calibrated probability" in result["uncertainty_definition"]


def test_next_best_evidence_engine():
    """Verify EVOI ranking and dynamic state updating."""
    engine = NextBestEvidenceEngine(policy="balanced")
    sample_fusion = {
        "incident_id": "TEST-02",
        "decision": "AMBIGUOUS_LEADS",
        "ranked_leads": [
            {"vessel_name": "CANDIDATE_A", "fusion_score": 0.58},
            {"vessel_name": "CANDIDATE_B", "fusion_score": 0.56},
        ],
    }

    rec = engine.evaluate_incident_sufficiency(sample_fusion)
    assert rec["current_evidence_sufficient"] is False
    assert len(rec["ranked_acquisition_options"]) > 0

    # Test recalculation with confirmed satellite-AIS gap-fill
    updated = engine.recalculate_with_new_evidence(
        initial_fusion=sample_fusion,
        acquired_modality_id="COMMERCIAL_SATELLITE_AIS",
        acquired_finding="Confirmed CANDIDATE_A transponder was manually switched off while loitering at spill site.",
    )
    assert updated["updated_decision"] == "ADDITIONAL_EVIDENCE_REVIEW_REQUIRED"
    assert updated["updated_top_candidate"] == "CANDIDATE_A"
    assert updated["updated_uncertainty"] == rec["current_incident_uncertainty"]
    assert updated["updated_ranked_leads"] == sample_fusion["ranked_leads"]


def test_cryptographic_audit_chain():
    """Verify SHA-256 hash chaining and tamper-detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = AuditTrail(log_dir=tmpdir)
        trail.log_event("TEST_EVENT_1", {"sensor": "SENTINEL-1", "chip": 101})
        trail.log_event("TEST_EVENT_2", {"fusion_score": 0.88, "verdict": "FLAGGED"})

        # Chain should verify
        valid, err = trail.verify_integrity()
        assert valid is True
        assert err is None

        # Simulate adversarial post-hoc tampering of block 1
        trail.chain[1].details["sensor"] = "TAMPERED_RADAR_NAME"
        tampered_valid, tamper_err = trail.verify_integrity()
        assert tampered_valid is False
        assert "mismatch" in tamper_err.lower()
