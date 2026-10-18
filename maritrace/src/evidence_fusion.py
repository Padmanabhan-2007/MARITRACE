"""MARITRACE Evidence-Fusion Engine.

Combines satellite SAR detections, physical drift hindcast envelopes,
reconstructed AIS trajectories, and machine learning vessel classifications
into ranked investigative leads with spatiotemporal gating, uncertainty awareness,
and mandatory abstention.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .audit_logger import AuditTrail


class SpatiotemporalValidationGate:
    """Pre-fusion validation gate enforcing temporal alignment and geographic containment."""

    def __init__(self, max_temporal_gap_hours: float = 24.0):
        self.max_temporal_gap_hours = max_temporal_gap_hours

    def validate(
        self,
        incident_time: pd.Timestamp,
        ais_start_time: pd.Timestamp,
        ais_end_time: pd.Timestamp,
        incident_lat: float,
        incident_lon: float,
        roi_lat: Tuple[float, float],
        roi_lon: Tuple[float, float],
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate whether AIS observations and incident spatiotemporally align."""
        if pd.isna(incident_time) or pd.isna(ais_start_time) or pd.isna(ais_end_time) or ais_start_time > ais_end_time:
            return False, "INVALID_SOURCE_TIMESTAMPS", {
                "reason": "Incident or AIS source timestamps are missing, invalid, or reversed.",
                "temporal_gap_hours": None,
            }
        # 1. Geographic Check
        in_lat = roi_lat[0] <= incident_lat <= roi_lat[1]
        in_lon = roi_lon[0] <= incident_lon <= roi_lon[1]
        if not (in_lat and in_lon):
            return (
                False,
                "GEOGRAPHIC_BOUNDS_MISMATCH",
                {
                    "reason": f"Incident location ({incident_lat}, {incident_lon}) outside AIS ROI {roi_lat}, {roi_lon}",
                    "temporal_gap_hours": None,
                },
            )

        # 2. Temporal Alignment Check
        # Check distance from incident time to closest AIS boundary
        if incident_time < ais_start_time:
            gap_sec = (ais_start_time - incident_time).total_seconds()
        elif incident_time > ais_end_time:
            gap_sec = (incident_time - ais_end_time).total_seconds()
        else:
            gap_sec = 0.0

        gap_hours = gap_sec / 3600.0

        if gap_hours > self.max_temporal_gap_hours:
            return (
                False,
                "TEMPORAL_WINDOW_MISMATCH",
                {
                    "reason": (
                        f"Temporal validation failure: AIS observation window ({ais_start_time.strftime('%Y-%m-%d')} to "
                        f"{ais_end_time.strftime('%Y-%m-%d')}) and incident time ({incident_time.strftime('%Y-%m-%d')}) "
                        f"differ by {gap_hours:.1f} hours (> {self.max_temporal_gap_hours:.1f}h allowed). "
                        "Vessels must be retained strictly as a Regional Historical Baseline Scenario."
                    ),
                    "temporal_gap_hours": round(gap_hours, 2),
                },
            )

        return True, "SPATIOTEMPORAL_ALIGNMENT_VERIFIED", {"temporal_gap_hours": round(gap_hours, 2)}


class EvidenceFusionEngine:
    """Uncertainty-aware evidence fusion engine with explicit abstention logic."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold_recommend_inspection: float = 0.65,
        threshold_abstain_score: float = 0.40,
        threshold_sar_min: float = 0.50,
        max_temporal_gap_hours: float = 24.0,
        audit_trail: Optional[AuditTrail] = None,
    ):
        self.weights = weights or {
            "spatial_drift_overlap": 0.35,
            "temporal_alignment": 0.25,
            "vessel_physical_profile": 0.20,
            "ais_anomaly": 0.20,
        }
        self.thresh_inspect = threshold_recommend_inspection
        self.thresh_abstain = threshold_abstain_score
        self.thresh_sar_min = threshold_sar_min
        self.gate = SpatiotemporalValidationGate(max_temporal_gap_hours=max_temporal_gap_hours)
        self.audit = audit_trail or AuditTrail()

    def fuse_incident_evidence(
        self,
        incident_id: str,
        incident_name: str,
        spill_lat: float,
        spill_lon: float,
        sar_oil_prob: Optional[float],
        drift_steps: List[Dict[str, Any]],
        vessel_trajectories: pd.DataFrame,
        incident_time_utc: str = "2024-01-07 10:00:00",
        allow_historical_baseline_mode: bool = True,
    ) -> Dict[str, Any]:
        """Perform multi-source evidence fusion with automatic pre-fusion gating."""
        print(f"\n[Evidence Fusion] Initiating forensic fusion for Incident {incident_id}: {incident_name}")
        print(
            "  SAR Oil Slick Probability: "
            + (f"{sar_oil_prob:.4f}" if sar_oil_prob is not None else "unavailable")
        )
        print(f"  Candidate vessels available: {len(vessel_trajectories)}")

        # Test-set metrics and training-chip scores are not incident-scene evidence.
        # Require an explicit score from a scene linked to this incident before ranking.
        try:
            sar_score = float(sar_oil_prob) if sar_oil_prob is not None else None
        except (TypeError, ValueError):
            sar_score = None
        valid_sar_score = sar_score is not None and math.isfinite(sar_score) and 0.0 <= sar_score <= 1.0
        if not valid_sar_score:
            decision = {
                "incident_id": incident_id,
                "incident_name": incident_name,
                "decision": "ABSTAIN",
                "status": "MISSING_INCIDENT_SAR" if sar_oil_prob is None else "INVALID_INCIDENT_SAR_SCORE",
                "sar_oil_prob": sar_score if sar_score is not None and math.isfinite(sar_score) else None,
                "primary_reason": (
                    "No valid incident-specific SAR score was supplied. Model evaluation metrics and training chips "
                    "cannot stand in for an observation of the selected incident scene."
                ),
                "ranked_leads": [],
                "candidate_count": 0,
                "forensic_weights": self.weights,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.audit.log_event("EVIDENCE_FUSION_ABSTENTION", decision)
            return decision
        sar_oil_prob = sar_score

        # Rule 1: Abstention on SAR false positive / lookalike
        if sar_oil_prob < self.thresh_sar_min:
            decision = {
                "incident_id": incident_id,
                "incident_name": incident_name,
                "decision": "ABSTAIN",
                "status": "NOT_AN_OIL_SLICK",
                "primary_reason": (
                    f"Satellite SAR classifier confidence ({sar_oil_prob:.3f}) is below minimum threshold "
                    f"({self.thresh_sar_min:.3f}). Feature classified as oceanic look-alike (wind calm/biogenic sheen)."
                ),
                "ranked_leads": [],
                "forensic_weights": self.weights,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.audit.log_event("EVIDENCE_FUSION_ABSTENTION", decision)
            return decision

        if vessel_trajectories.empty:
            decision = {
                "incident_id": incident_id,
                "incident_name": incident_name,
                "decision": "ABSTAIN",
                "status": "NO_VESSEL_DATA",
                "primary_reason": "No AIS vessel observations available in incident geographic area.",
                "ranked_leads": [],
                "forensic_weights": self.weights,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.audit.log_event("EVIDENCE_FUSION_ABSTENTION", decision)
            return decision

        # 2. Spatiotemporal Gating
        try:
            inc_dt = pd.to_datetime(incident_time_utc, utc=True)
        except (TypeError, ValueError):
            inc_dt = pd.NaT
        try:
            ais_start = pd.to_datetime(vessel_trajectories["start_time_utc"].min(), utc=True)
            ais_end = pd.to_datetime(vessel_trajectories["end_time_utc"].max(), utc=True)
        except (KeyError, TypeError, ValueError):
            ais_start, ais_end = pd.NaT, pd.NaT
        coverage_cols = ("coverage_min_lat", "coverage_max_lat", "coverage_min_lon", "coverage_max_lon")
        coverage_available = all(c in vessel_trajectories.columns for c in coverage_cols)
        if coverage_available:
            ais_roi_lat = (
                float(vessel_trajectories["coverage_min_lat"].min()),
                float(vessel_trajectories["coverage_max_lat"].max()),
            )
            ais_roi_lon = (
                float(vessel_trajectories["coverage_min_lon"].min()),
                float(vessel_trajectories["coverage_max_lon"].max()),
            )
        else:
            ais_roi_lat, ais_roi_lon = (1.0, -1.0), (1.0, -1.0)

        gate_passed, gate_code, gate_meta = self.gate.validate(
            incident_time=inc_dt,
            ais_start_time=ais_start,
            ais_end_time=ais_end,
            incident_lat=spill_lat,
            incident_lon=spill_lon,
            roi_lat=ais_roi_lat,
            roi_lon=ais_roi_lon,
        )

        is_baseline_scenario = not gate_passed and gate_code == "TEMPORAL_WINDOW_MISMATCH"

        # Historical regional AIS can describe traffic patterns, but must never
        # become incident-specific ranked leads.
        if not gate_passed:
            columns = [c for c in ("mmsi", "vessel_name", "start_time_utc", "end_time_utc")
                       if c in vessel_trajectories.columns]
            decision = {
                "incident_id": incident_id,
                "incident_name": incident_name,
                "incident_location": {"lat": spill_lat, "lon": spill_lon},
                "sar_oil_prob": round(sar_oil_prob, 4),
                "spatiotemporal_gate": {
                    "passed": False, "code": gate_code, "meta": gate_meta,
                    "ais_observation_window_utc": {
                    "start": ais_start.isoformat() if not pd.isna(ais_start) else None,
                    "end": ais_end.isoformat() if not pd.isna(ais_end) else None,
                    },
                    "incident_time_utc": inc_dt.isoformat(),
                },
                "decision": "ABSTAIN",
                "primary_reason": (
                    "AIS time window does not align with incident time; tracks are retained only as a regional historical baseline."
                    if is_baseline_scenario else
                    f"AIS validation failed ({gate_code}); no incident-specific vessel ranking is permitted."
                ),
                "is_historical_baseline": is_baseline_scenario,
                "candidate_count": 0,
                "ranked_leads": [],
                "historical_baseline_vessels": (
                    vessel_trajectories[columns].to_dict(orient="records") if is_baseline_scenario else []
                ),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.audit.log_event("EVIDENCE_FUSION_BASELINE_ONLY", decision)
            return decision

        # Extract origin envelope centroid and radius at 6 hours back
        drift_step_6h = next((s for s in drift_steps if s.get("hours_back") == 6.0), drift_steps[0] if drift_steps else {})
        envelope_lat = drift_step_6h.get("mean_lat", spill_lat)
        envelope_lon = drift_step_6h.get("mean_lon", spill_lon)
        envelope_radius_km = drift_step_6h.get("uncertainty_radius_km", 5.0)

        leads = []
        for _, vessel in vessel_trajectories.iterrows():
            # Dataset-wide overlap is not sufficient: each vessel needs an
            # incident-aligned observation window of its own.
            try:
                vessel_start = pd.to_datetime(vessel["start_time_utc"], utc=True)
                vessel_end = pd.to_datetime(vessel["end_time_utc"], utc=True)
                if pd.isna(vessel_start) or pd.isna(vessel_end) or vessel_start > vessel_end:
                    continue
                vessel_gap_hours = max(
                    (vessel_start - inc_dt).total_seconds() / 3600.0,
                    (inc_dt - vessel_end).total_seconds() / 3600.0,
                    0.0,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if vessel_gap_hours > self.gate.max_temporal_gap_hours:
                continue
            mmsi = str(vessel["mmsi"])
            v_name = str(vessel.get("vessel_name", "UNKNOWN"))
            raw_vtype = vessel.get("vessel_type")
            v_type = 0 if pd.isna(raw_vtype) else int(float(raw_vtype))
            min_dist = float(vessel.get("min_distance_to_spill_km", 999.0))
            gap_hrs = float(vessel.get("max_ais_gap_hours", 0.0))
            loiter_frac = float(vessel.get("loiter_fraction", 0.0))
            hours_from_window = float(vessel.get("hours_from_spill_window", 0.0))

            # Independently sourced physical vessel dimensions (length, width, draft)
            length = float(vessel.get("length", 0) or 0)
            width = float(vessel.get("width", 0) or 0)
            draft = float(vessel.get("draft", 0) or 0)

            # 1. Spatial Component: exponential decay relative to origin envelope radius
            spatial_feature = math.exp(-max(0.0, min_dist - envelope_radius_km) / 8.0)

            # 2. Temporal Component: decay over transit time offset
            if is_baseline_scenario:
                # In baseline scenario, evaluate temporal transit consistency relative to observation day
                temporal_feature = 0.50  # Neutral baseline anchor
            else:
                temporal_feature = math.exp(-hours_from_window / 6.0)

            # 3. Physical Vessel Profile (Derived from actual dimensions, not gear alone)
            # Physical displacement volume proxy: V = length * width * draft
            if length > 0 and width > 0:
                d_eff = draft if draft > 0 else 3.0
                vol_proxy = length * width * d_eff
                # Normalize typical commercial scale (1,000 m3 small tug to 150,000 m3 tanker)
                vessel_physical_feature = min(1.0, max(0.2, math.log10(max(10.0, vol_proxy)) / 5.5))
            else:
                # Fallback to broad IMO category baseline if physical dimensions missing
                vessel_physical_feature = 0.50

            # 4. AIS Anomaly & Gaps:
            # Blackout > 1 hr or loitering near spill increases anomaly score
            gap_score = min(1.0, gap_hrs / 4.0)
            anomaly_feature = 0.6 * gap_score + 0.4 * loiter_frac

            # Composite weighted fusion score
            w_spat = self.weights["spatial_drift_overlap"]
            w_temp = self.weights["temporal_alignment"]
            w_phys = self.weights["vessel_physical_profile"]
            w_anom = self.weights["ais_anomaly"]

            total_score = (
                w_spat * spatial_feature
                + w_temp * temporal_feature
                + w_phys * vessel_physical_feature
                + w_anom * anomaly_feature
            )

            # Multiply by SAR detection confidence
            calibrated_score = total_score * sar_oil_prob

            leads.append({
                "mmsi": mmsi,
                "vessel_name": v_name,
                "vessel_type": v_type,
                "dimensions": {"length_m": length, "width_m": width, "draft_m": draft},
                "fusion_score": round(float(calibrated_score), 4),
                "unweighted_score": round(float(total_score), 4),
                "factor_breakdown": {
                    "spatial_overlap": round(float(spatial_feature), 4),
                    "temporal_alignment": round(float(temporal_feature), 4),
                    "vessel_physical_size": round(float(vessel_physical_feature), 4),
                    "ais_anomaly": round(float(anomaly_feature), 4),
                },
                "observations": {
                    "min_distance_km": round(min_dist, 2),
                    "envelope_radius_km": round(envelope_radius_km, 2),
                    "ais_gap_hours": round(gap_hrs, 2),
                    "loiter_fraction": round(loiter_frac, 3),
                    "hours_from_window": round(hours_from_window, 2),
                    "ais_observation_start_utc": vessel_start.isoformat(),
                    "ais_observation_end_utc": vessel_end.isoformat(),
                    "incident_to_vessel_window_gap_hours": round(vessel_gap_hours, 2),
                },
                "audit_note": (
                    "HISTORICAL_BASELINE_OBSERVATION: Proximity within historical baseline does NOT constitute "
                    "evidence of liability for a subsequent discharge."
                    if is_baseline_scenario
                    else "Observational evidence only. Proximity does not constitute proof of liability."
                ),
            })

        leads = sorted(leads, key=lambda x: x["fusion_score"], reverse=True)

        # Decision & Abstention Logic
        top_lead = leads[0] if leads else None

        if is_baseline_scenario:
            decision_status = "ABSTAIN"
            decision_reason = (
                f"TEMPORAL_WINDOW_MISMATCH: Available AIS tracking (2024-01-02) does not overlap the reported "
                f"incident date ({inc_dt.strftime('%Y-%m-%d')}, gap = {gate_meta['temporal_gap_hours']}h > 24h). "
                "Candidate vessels are cataloged strictly as a Regional Historical Traffic Baseline. "
                "System abstains from suspect attribution to protect innocent mariners."
            )
        elif top_lead is None or top_lead["fusion_score"] < self.thresh_abstain:
            decision_status = "ABSTAIN"
            decision_reason = (
                f"Top candidate fusion score ({top_lead['fusion_score'] if top_lead else 0:.3f}) "
                f"falls below minimum threshold ({self.thresh_abstain:.3f}). Evidence is inconclusive."
            )
        elif len(leads) >= 2 and abs(leads[0]["fusion_score"] - leads[1]["fusion_score"]) < 0.04:
            decision_status = "AMBIGUOUS_LEADS"
            decision_reason = (
                f"Top candidates {leads[0]['vessel_name']} ({leads[0]['fusion_score']:.3f}) and "
                f"{leads[1]['vessel_name']} ({leads[1]['fusion_score']:.3f}) have indistinguishable scores. "
                "Additional evidence acquisition required to discriminate source."
            )
        else:
            decision_status = "ACTIONABLE_LEAD"
            decision_reason = "Sufficient spatial-temporal corroboration to recommend formal port inspection."

        fusion_result = {
            "incident_id": incident_id,
            "incident_name": incident_name,
            "incident_location": {"lat": spill_lat, "lon": spill_lon},
            "sar_oil_prob": round(sar_oil_prob, 4),
            "spatiotemporal_gate": {
                "passed": gate_passed,
                "code": gate_code,
                "meta": gate_meta,
            },
            "decision": decision_status,
            "primary_reason": decision_reason,
            "is_historical_baseline": is_baseline_scenario,
            "top_candidate": top_lead["vessel_name"] if top_lead and decision_status == "ACTIONABLE_LEAD" else None,
            "candidate_count": len(leads),
            "ranked_leads": leads[:10],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.audit.log_event("EVIDENCE_FUSION_COMPLETED", {
            "incident_id": incident_id,
            "decision": decision_status,
            "gate_code": gate_code,
            "is_historical_baseline": is_baseline_scenario,
            "top_vessel": top_lead["vessel_name"] if top_lead else "NONE",
        })

        return fusion_result
