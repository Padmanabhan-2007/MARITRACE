"""Coordinate validation helpers for dashboard map layers."""
from __future__ import annotations

import pandas as pd


def valid_coordinate_rows(frame: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Return rows with numeric WGS84 latitude/longitude in valid ranges."""
    if lat_col not in frame or lon_col not in frame:
        return frame.iloc[0:0].copy()
    result = frame.copy()
    result[lat_col] = pd.to_numeric(result[lat_col], errors="coerce")
    result[lon_col] = pd.to_numeric(result[lon_col], errors="coerce")
    valid = result[lat_col].between(-90, 90) & result[lon_col].between(-180, 180)
    return result.loc[valid].copy()

