"""MARITRACE Ocean Current and Physically Grounded Drift Estimation Engine.

Validates input NetCDF hydrodynamic data (gracefully handling empty/corrupted files),
computes Lagrangian advection with Coriolis wind deflection, turbulent eddy diffusion,
and produces backward (origin hindcast) and forward (dispersion forecast) envelopes
as GeoJSON and CSV.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OceanDriftEngine:
    """Lagrangian oil drift simulation engine combining currents, wind, and turbulent diffusion."""

    def __init__(
        self,
        wind_factor: float = 0.03,
        coriolis_deflection_deg: float = 12.0,
        eddy_diffusivity_m2s: float = 15.0,
    ):
        self.wind_factor = wind_factor
        self.coriolis_deflection_rad = math.radians(coriolis_deflection_deg)
        self.eddy_diffusivity = eddy_diffusivity_m2s

    @staticmethod
    def validate_netcdf(filepath: str) -> Dict[str, Any]:
        """Validate NetCDF integrity, checking for 0-byte or corrupted files."""
        if not os.path.exists(filepath):
            return {
                "valid": False,
                "status": "MISSING_FILE",
                "reason": f"File does not exist: {filepath}",
                "size_bytes": 0,
            }

        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            return {
                "valid": False,
                "status": "CORRUPTED_ZERO_BYTES",
                "reason": "File exists on disk but has size 0 bytes. NetCDF header missing.",
                "size_bytes": 0,
            }

        try:
            import netCDF4
            with netCDF4.Dataset(filepath, "r") as ds:
                variables = list(ds.variables.keys())
                return {
                    "valid": True,
                    "status": "VALID",
                    "variables": variables,
                    "size_bytes": size_bytes,
                }
        except Exception as exc:
            return {
                "valid": False,
                "status": "UNREADABLE_NETCDF",
                "reason": f"{type(exc).__name__}: {str(exc)}",
                "size_bytes": size_bytes,
            }

    def compute_drift_velocity(
        self,
        cur_u: float,
        cur_v: float,
        wind_u: float,
        wind_v: float,
    ) -> Tuple[float, float]:
        """Compute combined drift velocity vector (m/s) with Coriolis deflection.

        In the Northern Hemisphere, surface wind drift is deflected ~10-15 degrees
        to the right of the wind direction (Ekman spiral effect).
        """
        # Wind drift magnitude and rotated components
        cos_c = math.cos(self.coriolis_deflection_rad)
        sin_c = math.sin(self.coriolis_deflection_rad)

        wind_drift_u = self.wind_factor * (wind_u * cos_c + wind_v * sin_c)
        wind_drift_v = self.wind_factor * (-wind_u * sin_c + wind_v * cos_c)

        total_u = cur_u + wind_drift_u
        total_v = cur_v + wind_drift_v
        return total_u, total_v

    def simulate_back_drift(
        self,
        spill_lat: float,
        spill_lon: float,
        hours_list: List[float] = [2, 4, 6, 12, 18, 24],
        cur_u: float = 0.18,
        cur_v: float = -0.08,
        wind_u: float = -3.5,
        wind_v: float = -2.0,
        n_particles: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Perform backward Lagrangian Monte Carlo hindcast to reconstruct origin envelopes."""
        np.random.seed(seed)
        total_u, total_v = self.compute_drift_velocity(cur_u, cur_v, wind_u, wind_v)

        # Standard conversion meters to degrees
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * max(0.01, math.cos(math.radians(spill_lat)))

        trajectory_steps = []
        geojson_features = []

        max_hours = max(hours_list)
        dt_sec = 1800  # 30-minute numerical integration steps
        total_steps = int(max_hours * 3600 / dt_sec)

        # Initialize particles at spill center
        particles_lat = np.full(n_particles, spill_lat)
        particles_lon = np.full(n_particles, spill_lon)

        current_hour = 0.0
        step_idx = 0

        target_hours_set = sorted(hours_list)

        while current_hour <= max_hours + 1e-4:
            if any(abs(current_hour - h) < (dt_sec / 7200.0) for h in target_hours_set):
                mean_lat = float(np.mean(particles_lat))
                mean_lon = float(np.mean(particles_lon))
                std_lat = float(np.std(particles_lat))
                std_lon = float(np.std(particles_lon))

                # Diffusion radius in km (approx 2 sigma + 2km baseline buffer)
                uncertainty_radius_km = max(2.0, 2.0 * math.sqrt(2 * self.eddy_diffusivity * (current_hour * 3600)) / 1000.0)

                trajectory_steps.append({
                    "hours_back": current_hour,
                    "mean_lat": round(mean_lat, 6),
                    "mean_lon": round(mean_lon, 6),
                    "uncertainty_radius_km": round(uncertainty_radius_km, 2),
                    "cur_u": cur_u,
                    "cur_v": cur_v,
                    "wind_u": wind_u,
                    "wind_v": wind_v,
                    "drift_speed_knots": round(math.sqrt(total_u**2 + total_v**2) * 1.94384, 3),
                })

                # Construct circle polygon in GeoJSON format
                polygon_coords = []
                for deg in range(0, 361, 15):
                    rad = math.radians(deg)
                    d_north = uncertainty_radius_km * 1000.0 * math.sin(rad)
                    d_east = uncertainty_radius_km * 1000.0 * math.cos(rad)
                    p_lat = mean_lat + d_north / m_per_deg_lat
                    p_lon = mean_lon + d_east / m_per_deg_lon
                    polygon_coords.append([round(p_lon, 6), round(p_lat, 6)])

                geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "hours_back": current_hour,
                        "mean_lat": mean_lat,
                        "mean_lon": mean_lon,
                        "radius_km": uncertainty_radius_km,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords],
                    },
                })

            # Negative velocity for backward integration
            d_east_mean = -total_u * dt_sec
            d_north_mean = -total_v * dt_sec

            # Stochastic turbulent eddy diffusion
            diffusion_scale = math.sqrt(2.0 * self.eddy_diffusivity * dt_sec)
            rand_east = np.random.normal(0, diffusion_scale, n_particles)
            rand_north = np.random.normal(0, diffusion_scale, n_particles)

            particles_lat += (d_north_mean + rand_north) / m_per_deg_lat
            particles_lon += (d_east_mean + rand_east) / m_per_deg_lon

            current_hour += dt_sec / 3600.0
            step_idx += 1

        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features,
        }

        return {
            "trajectory_steps": trajectory_steps,
            "geojson": geojson_data,
        }


def run_drift_pipeline(
    netcdf_path: str = r"D:\dataset maritrace\cmems_obs-mob_glo_phy-cur_nrt_0.25deg_P1D-m_1792237857475.nc",
    output_dir: str = r"D:\dataset maritrace\outputs\processed_data",
    plots_dir: str = r"D:\dataset maritrace\outputs\plots",
    spill_lat: float = 29.1350,
    spill_lon: float = -89.3039,
) -> Dict[str, Any]:
    """Execute complete ocean drift validation, simulation, and export."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("MARITRACE OCEAN HYDRODYNAMICS AND DRIFT FORENSICS")
    print("=" * 60)

    # 1. NetCDF Validation
    print(f"Validating ocean current NetCDF: {netcdf_path}")
    nc_status = OceanDriftEngine.validate_netcdf(netcdf_path)
    print(f"  NetCDF Status: {nc_status['status']}")
    if not nc_status["valid"]:
        print(f"  REASON: {nc_status.get('reason')}")
        print("  NOTICE: No valid incident-matched environmental data was loaded; using illustrative scenario parameters.")
        provenance_mode = "SCENARIO_ESTIMATE_NO_VALIDATED_HINDCAST"
        cur_u = 0.18   # m/s eastward
        cur_v = -0.08  # m/s southward
        wind_u = -3.5  # m/s westward (easterly wind)
        wind_v = -2.0  # m/s southward
    else:
        # Validation only confirms that a file opens. This pipeline does not
        # currently extract gridded values from it, so these remain scenarios.
        provenance_mode = "SCENARIO_ESTIMATE_NETCDF_NOT_INGESTED"
        cur_u = 0.18
        cur_v = -0.08
        wind_u = -3.5
        wind_v = -2.0

    # 2. Run Drift Simulation
    engine = OceanDriftEngine(wind_factor=0.03, coriolis_deflection_deg=12.0, eddy_diffusivity_m2s=15.0)
    sim_result = engine.simulate_back_drift(
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        hours_list=[2, 4, 6, 12, 18, 24],
        cur_u=cur_u,
        cur_v=cur_v,
        wind_u=wind_u,
        wind_v=wind_v,
    )

    steps_df = pd.DataFrame(sim_result["trajectory_steps"])
    steps_df["provenance_mode"] = provenance_mode

    # Sensitivity cases are explicit scenario perturbations, not confidence
    # intervals and not substitutes for an incident matched hindcast.
    sensitivity_specs = [
        {"name": "low_advection", "current_scale": 0.5, "wind_factor": 0.01, "diffusivity": 5.0},
        {"name": "central_scenario", "current_scale": 1.0, "wind_factor": 0.03, "diffusivity": 15.0},
        {"name": "high_advection", "current_scale": 1.5, "wind_factor": 0.05, "diffusivity": 40.0},
    ]
    sensitivity = []
    for spec in sensitivity_specs:
        scenario_engine = OceanDriftEngine(
            wind_factor=spec["wind_factor"], coriolis_deflection_deg=12.0,
            eddy_diffusivity_m2s=spec["diffusivity"],
        )
        scenario = scenario_engine.simulate_back_drift(
            spill_lat=spill_lat, spill_lon=spill_lon, hours_list=[6, 24],
            cur_u=cur_u * spec["current_scale"], cur_v=cur_v * spec["current_scale"],
            wind_u=wind_u, wind_v=wind_v, seed=42,
        )
        sensitivity.append({
            **spec,
            "steps": scenario["trajectory_steps"],
        })

    # 3. Export CSV and GeoJSON
    csv_path = os.path.join(output_dir, "drift_backward_trajectory.csv")
    geojson_path = os.path.join(output_dir, "drift_origin_envelopes.geojson")

    steps_df.to_csv(csv_path, index=False)
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(sim_result["geojson"], f, indent=2)
    sensitivity_path = os.path.join(output_dir, "drift_sensitivity_scenarios.json")
    with open(sensitivity_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": "SCENARIO_SENSITIVITY_NOT_A_VALIDATED_HINDCAST",
            "netcdf_validation": nc_status,
            "scenarios": sensitivity,
        }, f, indent=2)

    print(f"  Saved drift trajectory CSV to: {csv_path}")
    print(f"  Saved origin envelopes GeoJSON to: {geojson_path}")

    # 4. Generate Drift Simulation Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    lats = steps_df["mean_lat"]
    lons = steps_df["mean_lon"]
    radii = steps_df["uncertainty_radius_km"]

    # Plot origin envelopes as expanding circles
    for lat, lon, r in zip(lats, lons, radii):
        circle = plt.Circle((lon, lat), r / 111.0, color="orange", alpha=0.15, ec="darkorange", lw=1.5)
        ax.add_patch(circle)

    ax.plot(lons, lats, "r--o", lw=2, label="Back-Drift Hindcast Path")
    ax.scatter([spill_lon], [spill_lat], color="blue", s=100, zorder=5, label=f"Observed Slick ({spill_lat}, {spill_lon})")

    for idx, row in steps_df.iterrows():
        ax.annotate(
            f"-{row['hours_back']}h\n(±{row['uncertainty_radius_km']}km)",
            (row["mean_lon"], row["mean_lat"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )

    ax.set_title(f"MARITRACE Backward Drift Trajectory & Probable Origin Envelopes\nMode: {provenance_mode}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()

    map_path = os.path.join(plots_dir, "drift_hindcast_envelope.png")
    fig.savefig(map_path, dpi=300)
    plt.close(fig)
    print(f"  Saved drift hindcast plot to: {map_path}")

    return {
        "nc_validation": nc_status,
        "provenance_mode": provenance_mode,
        "trajectory_csv": csv_path,
        "origin_geojson": geojson_path,
        "sensitivity_scenarios": sensitivity_path,
        "plot_path": map_path,
        "steps": steps_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_drift_pipeline()
