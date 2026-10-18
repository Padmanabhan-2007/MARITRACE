"""MARITRACE AIS Trajectory and Spatiotemporal Forensics Engine.

Incrementally reads compressed AIS Zstandard streams, filters by geographic ROI
and temporal windows, normalizes schemas, and reconstructs vessel trajectories
with kinematic anomaly, loitering, and blackout gap detection.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two geographic coordinates in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def stream_filter_ais_zst(
    zst_path: str,
    output_csv_path: str,
    roi_lat: Tuple[float, float] = (28.5, 29.8),
    roi_lon: Tuple[float, float] = (-90.5, -88.5),
    max_records: int = 5000000,
    chunk_size: int = 50000,
) -> Tuple[int, int, str]:
    """Incrementally decompress and filter AIS observations inside a bounding box.

    Avoids decompressing multi-gigabyte files to disk.
    """
    if zstd is None:
        raise ImportError("zstandard library is required to stream .csv.zst")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    t0 = time.time()
    total_read = 0
    matched_records = 0

    print(f"\n[AIS Engine] Streaming and filtering from {zst_path}...")
    print(f"  Geographic ROI: Lat {roi_lat}, Lon {roi_lon}")

    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            csv_reader = csv.reader(text_stream)
            raw_header = next(csv_reader)

            # Map raw header names
            header_map = {col.strip().lower(): idx for idx, col in enumerate(raw_header)}
            lat_idx = header_map.get("latitude")
            lon_idx = header_map.get("longitude")

            if lat_idx is None or lon_idx is None:
                raise ValueError(f"Could not find latitude/longitude columns in {raw_header}")

            with open(output_csv_path, "w", newline="", encoding="utf-8") as out_fh:
                csv_writer = csv.writer(out_fh)
                csv_writer.writerow(raw_header)

                batch = []
                for row in csv_reader:
                    total_read += 1
                    try:
                        lat = float(row[lat_idx])
                        lon = float(row[lon_idx])
                        if (roi_lat[0] <= lat <= roi_lat[1]) and (roi_lon[0] <= lon <= roi_lon[1]):
                            batch.append(row)
                            matched_records += 1
                    except (ValueError, IndexError):
                        continue

                    if len(batch) >= chunk_size:
                        csv_writer.writerows(batch)
                        batch = []
                        if total_read % 500000 == 0:
                            print(f"  Processed {total_read:,} rows... Matched {matched_records:,} in ROI.")

                    if total_read >= max_records:
                        break

                if batch:
                    csv_writer.writerows(batch)

    elapsed = round(time.time() - t0, 2)
    print(f"  Filtering complete: Read {total_read:,} observations -> Retained {matched_records:,} in ROI ({elapsed}s).")
    print(f"  Saved filtered AIS observations to: {output_csv_path}")
    return total_read, matched_records, output_csv_path


def reconstruct_vessel_trajectories(
    filtered_csv_path: str,
    incident_lat: float = 29.1350,
    incident_lon: float = -89.3039,
    incident_time: str = "2024-01-02 12:00:00",
    output_summary_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Reconstruct trajectory statistics, distance to spill, and kinematic anomalies per vessel."""
    print(f"\n[AIS Engine] Reconstructing vessel trajectories from {filtered_csv_path}...")
    df = pd.read_csv(filtered_csv_path)
    if df.empty:
        print("  Warning: Filtered AIS dataset is empty.")
        return pd.DataFrame()

    # Standardize column names
    col_map = {
        "base_date_time": "timestamp_utc",
        "latitude": "lat",
        "longitude": "lon",
        "sog": "speed",
        "cog": "course",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Validate coordinates & types
    df = df.dropna(subset=["mmsi", "lat", "lon", "timestamp_utc"]).copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Sort strictly by vessel and time
    df = df.sort_values(["mmsi", "timestamp_utc"]).reset_index(drop=True)

    incident_dt = pd.to_datetime(incident_time, utc=True)

    # Group by MMSI and compute trajectory features
    vessel_summaries = []
    for mmsi, group in df.groupby("mmsi"):
        if len(group) < 2:
            continue

        times = group["timestamp_utc"].values
        lats = group["lat"].values
        lons = group["lon"].values
        speeds = group["speed"].values

        # Time gaps (hours)
        time_diffs_sec = np.diff(times).astype("timedelta64[s]").astype(float)
        time_gaps_hr = time_diffs_sec / 3600.0
        max_gap_hours = float(np.max(time_gaps_hr)) if len(time_gaps_hr) > 0 else 0.0

        # Distances to incident location
        distances_km = [haversine_km(lat, lon, incident_lat, incident_lon) for lat, lon in zip(lats, lons)]
        min_dist_km = float(np.min(distances_km))
        min_dist_idx = int(np.argmin(distances_km))
        time_at_min_dist = group.iloc[min_dist_idx]["timestamp_utc"]
        time_diff_to_incident_hr = abs((time_at_min_dist - incident_dt).total_seconds()) / 3600.0

        # Loitering (speed < 2.0 knots while far from port)
        loiter_count = np.sum(speeds < 2.0)
        loiter_fraction = float(loiter_count / len(speeds))

        # Speed statistics
        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))

        # Metadata from last record
        last_rec = group.iloc[-1]
        vessel_name = str(last_rec.get("vessel_name", "UNKNOWN")).strip()
        vessel_type = last_rec.get("vessel_type", 0)
        vessel_length = last_rec.get("length", 0)
        vessel_width = last_rec.get("width", 0)
        vessel_draft = last_rec.get("draft", 0)
        cargo_type = last_rec.get("cargo", 0)

        vessel_summaries.append({
            "mmsi": str(mmsi),
            "vessel_name": vessel_name,
            "vessel_type": vessel_type,
            "length": vessel_length,
            "width": vessel_width,
            "draft": vessel_draft,
            "cargo": cargo_type,
            "obs_count": len(group),
            "start_time_utc": str(group.iloc[0]["timestamp_utc"]),
            "end_time_utc": str(group.iloc[-1]["timestamp_utc"]),
            "coverage_min_lat": float(np.min(lats)),
            "coverage_max_lat": float(np.max(lats)),
            "coverage_min_lon": float(np.min(lons)),
            "coverage_max_lon": float(np.max(lons)),
            "min_distance_to_spill_km": round(min_dist_km, 2),
            "time_at_min_distance": str(time_at_min_dist),
            "hours_from_spill_window": round(time_diff_to_incident_hr, 2),
            "max_ais_gap_hours": round(max_gap_hours, 2),
            "mean_speed_knots": round(mean_speed, 2),
            "max_speed_knots": round(max_speed, 2),
            "loiter_fraction": round(loiter_fraction, 3),
        })

    traj_df = pd.DataFrame(vessel_summaries)
    if not traj_df.empty:
        traj_df = traj_df.sort_values("min_distance_to_spill_km").reset_index(drop=True)
        print(f"  Reconstructed trajectories for {len(traj_df)} distinct vessels.")
        print(f"  Top 3 closest vessels to spill ({incident_lat}, {incident_lon}):")
        for idx, row in traj_df.head(3).iterrows():
            print(f"    - MMSI: {row['mmsi']} ({row['vessel_name']}) | Dist: {row['min_distance_to_spill_km']} km | Gap: {row['max_ais_gap_hours']}h | Speed: {row['mean_speed_knots']} kts")

        if output_summary_csv:
            os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
            traj_df.to_csv(output_summary_csv, index=False)
            print(f"  Saved trajectory forensic summary to: {output_summary_csv}")

    return traj_df


if __name__ == "__main__":
    zst_file = r"D:\dataset maritrace\ais-2024-01-02.csv.zst"
    out_filtered = r"D:\dataset maritrace\outputs\processed_data\ais_filtered_gom.csv"
    out_traj = r"D:\dataset maritrace\outputs\processed_data\vessel_trajectories.csv"

    # Filter to Mississippi Delta / Gulf of Mexico ROI
    stream_filter_ais_zst(
        zst_path=zst_file,
        output_csv_path=out_filtered,
        roi_lat=(28.5, 29.8),
        roi_lon=(-90.2, -88.5),
        max_records=1500000,
    )

    reconstruct_vessel_trajectories(
        filtered_csv_path=out_filtered,
        incident_lat=29.1350,
        incident_lon=-89.3039,
        incident_time="2024-01-02 12:00:00",
        output_summary_csv=out_traj,
    )
