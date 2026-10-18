"""MARITRACE Evidence Sufficiency and Next-Best-Evidence Engine.

Evaluates evidentiary gaps, models Expected Value of Information (EVOI),
and recommends prioritized acquisition actions (satellite tasking, aerial surveillance,
commercial satellite-AIS, port state sampling) under configurable decision policies.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class CandidateEvidenceModalities:
    """Catalog of observable evidence modalities with cost, delay, and uncertainty properties."""

    MODALITIES = [
        {
            "id": "SAT_OPTICAL_HIGHRES",
            "name": "High-Resolution Optical Satellite Tasking (Sentinel-2 / PlanetScope)",
            "description": "Acquire multispectral optical imagery to verify rainbow sheen interference and rule out biogenic algae blooms.",
            "estimated_cost_usd": 250.0,
            "latency_hours": 12.0,
            "uncertainty_reduction": 0.25,
            "prerequisites": "Cloud cover forecast < 30% over coordinates.",
            "impact_on_decision": "High: Eliminates biogenic false positives and confirms mineral hydrocarbon presence.",
        },
        {
            "id": "COMMERCIAL_SATELLITE_AIS",
            "name": "Commercial Satellite-AIS Gap-Fill (Spire / exactEarth)",
            "description": "Query oceanic satellite-AIS archives to reconstruct vessel trajectories during terrestrial antenna blackouts.",
            "estimated_cost_usd": 500.0,
            "latency_hours": 2.0,
            "uncertainty_reduction": 0.35,
            "prerequisites": "MMSI candidate list and 24h temporal bounding window.",
            "impact_on_decision": "Very High: Resolves terrestrial AIS dropouts to determine if vessel deliberately turned off transponder.",
        },
        {
            "id": "AERIAL_PATROL_SURVEILLANCE",
            "name": "Maritime Patrol Aircraft Overflight (USCG / SLAR-FLIR)",
            "description": "Dispatch fixed-wing maritime patrol equipped with Side-Looking Airborne Radar and infrared/ultraviolet line scanners.",
            "estimated_cost_usd": 4500.0,
            "latency_hours": 4.0,
            "uncertainty_reduction": 0.45,
            "prerequisites": "Incident within aircraft operational radius and daylight/VFR conditions.",
            "impact_on_decision": "Decisive: Obtains photographic visual evidence of trailing discharge directly from suspect vessel stern.",
        },
        {
            "id": "PORT_STATE_PHYSICAL_SAMPLING",
            "name": "Port State Control Fuel & Bilge Forensic Sampling",
            "description": "Inspector boards vessel at next port of call to take fuel/bilge samples for Gas Chromatography-Mass Spectrometry (GC-MS).",
            "estimated_cost_usd": 1200.0,
            "latency_hours": 36.0,
            "uncertainty_reduction": 0.55,
            "prerequisites": "Confirmed destination port and berth schedule.",
            "impact_on_decision": "Definitive / Court Admissible: Provides chemical hydrocarbon biomarker match between slick and bilge tank.",
        },
        {
            "id": "HYDRODYNAMIC_REANALYSIS_HIGHRES",
            "name": "Copernicus Marine High-Resolution Reanalysis Hindcast",
            "description": "Download 1/12th degree physical ocean reanalysis with baroclinic tidal currents to refine drift cone.",
            "estimated_cost_usd": 0.0,
            "latency_hours": 8.0,
            "uncertainty_reduction": 0.15,
            "prerequisites": "Copernicus Marine Service operational data release window.",
            "impact_on_decision": "Moderate: Narrows probable spill origin cone from 8 km to 3 km.",
        },
    ]


class NextBestEvidenceEngine:
    """Decision-support engine for ranking evidence acquisition under explicit policy constraints."""

    POLICIES = {
        "balanced": {"weight_uncertainty": 0.50, "weight_cost": 0.25, "weight_speed": 0.25},
        "urgent_response": {"weight_uncertainty": 0.40, "weight_cost": 0.10, "weight_speed": 0.50},
        "budget_constrained": {"weight_uncertainty": 0.40, "weight_cost": 0.50, "weight_speed": 0.10},
        "court_burden_of_proof": {"weight_uncertainty": 0.70, "weight_cost": 0.15, "weight_speed": 0.15},
    }

    def __init__(self, policy: str = "balanced"):
        self.policy_name = policy if policy in self.POLICIES else "balanced"
        self.policy_weights = self.POLICIES[self.policy_name]

    def evaluate_incident_sufficiency(
        self,
        fusion_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess whether current evidence warrants an enforcement action or requires acquisition."""
        decision = fusion_result.get("decision", "ABSTAIN")
        leads = fusion_result.get("ranked_leads", [])

        # Current uncertainty calculation
        if decision == "ABSTAIN":
            current_uncertainty = 0.85
        elif decision == "AMBIGUOUS_LEADS":
            current_uncertainty = 0.65
        else:
            top_score = leads[0]["fusion_score"] if leads else 0.5
            current_uncertainty = round(max(0.15, 1.0 - top_score), 3)

        is_sufficient = (decision == "ACTIONABLE_LEAD") and (current_uncertainty <= 0.35)

        # Rank candidate acquisitions based on EVOI (Expected Value of Information)
        ranked_options = []
        w_unc = self.policy_weights["weight_uncertainty"]
        w_cost = self.policy_weights["weight_cost"]
        w_spd = self.policy_weights["weight_speed"]

        for item in CandidateEvidenceModalities.MODALITIES:
            unc_red = item["uncertainty_reduction"]
            cost = item["estimated_cost_usd"]
            lat_hr = item["latency_hours"]

            # Normalized components (0 to 1)
            norm_unc = unc_red / 0.60
            norm_cost = 1.0 - min(1.0, cost / 5000.0)
            norm_speed = 1.0 - min(1.0, lat_hr / 48.0)

            priority_score = w_unc * norm_unc + w_cost * norm_cost + w_spd * norm_speed
            evoi_ratio = round(unc_red / max(1.0, cost / 100.0), 3)

            # These are heuristic decrements on a normalized index, not Bayesian updates.
            posterior_u = round(max(0.05, current_uncertainty - unc_red), 3)
            actual_decrement = round(current_uncertainty - posterior_u, 3)
            rel_reduction_pct = round(
                actual_decrement / current_uncertainty * 100 if current_uncertainty else 0.0, 1
            )

            ranked_options.append({
                "id": item["id"],
                "name": item["name"],
                "description": item["description"],
                "priority_score": round(float(priority_score), 4),
                "absolute_uncertainty_reduction": actual_decrement,
                "requested_heuristic_uncertainty_decrement": unc_red,
                "relative_uncertainty_reduction_pct": rel_reduction_pct,
                "prior_uncertainty": current_uncertainty,
                "projected_posterior_uncertainty": posterior_u,
                "reduction_summary": (
                    f"{unc_red*100:.0f}% absolute reduction on [0, 1] scale "
                    f"(from {current_uncertainty*100:.1f}% to {posterior_u*100:.1f}%, "
                    f"a {rel_reduction_pct}% relative decrease)"
                ),
                "cost_usd": cost,
                "latency_hours": lat_hr,
                "evoi_index": evoi_ratio,
                "prerequisites": item["prerequisites"],
                "impact": item["impact_on_decision"],
            })

        ranked_options = sorted(ranked_options, key=lambda x: x["priority_score"], reverse=True)

        recommendation = {
            "incident_id": fusion_result.get("incident_id"),
            "current_evidence_sufficient": is_sufficient,
            "current_incident_uncertainty": current_uncertainty,
            "decision_policy": self.policy_name,
            "policy_weights": self.policy_weights,
            "scoring_formula": "Priority = w_unc * (ΔU/0.60) + w_cost * (1 - Cost/5000) + w_spd * (1 - Latency/48)",
            "uncertainty_definition": "Heuristic decision-support index from 0 (low uncertainty) to 1 (high uncertainty); not a calibrated probability or Bayesian posterior.",
            "methodology_disclosure": (
                "Acquisition decrements are illustrative heuristics, not measured information gain, "
                "calibrated Bayesian posterior updates, or empirically validated operational parameters. "
                "Relative reduction is computed from the capped posterior: (prior - posterior) / prior."
            ),
            "primary_recommendation": ranked_options[0] if not is_sufficient else None,
            "ranked_acquisition_options": ranked_options,
            "guidance_notes": (
                "If current evidence is insufficient, proceed with top-ranked acquisition option. "
                "Recalculate state and suspect rankings once new evidence is ingested."
            ),
        }
        return recommendation

    def recalculate_with_new_evidence(
        self,
        initial_fusion: Dict[str, Any],
        acquired_modality_id: str,
        acquired_finding: str,
    ) -> Dict[str, Any]:
        """Record a finding for review without treating free text as validated evidence."""
        modality = next((m for m in CandidateEvidenceModalities.MODALITIES if m["id"] == acquired_modality_id), None)
        if not modality:
            raise ValueError(f"Unknown modality ID: {acquired_modality_id}")

        # A free-text note is not a validated evidence object. Do not infer
        # confirmation, exculpation, or new candidate scores from keyword matches.
        leads = initial_fusion.get("ranked_leads", []).copy()
        prior_uncertainty = initial_fusion.get("current_incident_uncertainty")
        if prior_uncertainty is None:
            prior_uncertainty = 0.85 if initial_fusion.get("decision") == "ABSTAIN" else 0.65
        updated_decision = "ADDITIONAL_EVIDENCE_REVIEW_REQUIRED"
        updated_reason = (
            f"A note was recorded for {modality['name']}, but its source and contents have not been "
            "independently validated. Keep the existing lead scores unchanged and review the evidence before updating any decision."
        )
        new_uncertainty = prior_uncertainty

        return {
            "incident_id": initial_fusion.get("incident_id"),
            "evidence_added": {
                "modality_id": acquired_modality_id,
                "modality_name": modality["name"],
                "finding": acquired_finding,
            },
            "updated_decision": updated_decision,
            "updated_reason": updated_reason,
            "updated_uncertainty": new_uncertainty,
            "updated_top_candidate": leads[0]["vessel_name"] if leads and updated_decision != "ABSTAIN" else None,
            "updated_ranked_leads": leads[:5],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


if __name__ == "__main__":
    engine = NextBestEvidenceEngine(policy="balanced")
    sample_fusion = {
        "incident_id": "10716",
        "decision": "AMBIGUOUS_LEADS",
        "ranked_leads": [
            {"vessel_name": "MISS GWEN", "fusion_score": 0.62},
            {"vessel_name": "CAPT PHILLIP BOX", "fusion_score": 0.60},
        ],
    }
    rec = engine.evaluate_incident_sufficiency(sample_fusion)
    print("Top Recommended Evidence to Acquire:")
    print(f"  {rec['primary_recommendation']['name']} (Priority: {rec['primary_recommendation']['priority_score']})")
