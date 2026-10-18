import json

import pandas as pd

from src.dashboard_data import load_csv, load_json
from src.dashboard_maps import valid_coordinate_rows
from src.evidence_fusion import SpatiotemporalValidationGate
import pandas as pd


def test_dashboard_loaders_handle_missing_empty_and_malformed_files(tmp_path):
    assert load_csv(tmp_path / "missing.csv").empty
    assert load_json(tmp_path / "missing.json") is None

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert load_csv(empty).empty

    malformed = tmp_path / "bad.json"
    malformed.write_text("{")
    assert load_json(malformed) is None


def test_dashboard_csv_loader_limits_rows(tmp_path):
    path = tmp_path / "observations.csv"
    pd.DataFrame({"mmsi": [1, 2, 3], "lat": [0.0, 1.0, 2.0]}).to_csv(path, index=False)
    result = load_csv(path, nrows=2)
    assert len(result) == 2
    assert result.columns.tolist() == ["mmsi", "lat"]


def test_dashboard_json_loader_reads_recorded_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    expected = {"test_metrics": {"accuracy": 0.94}}
    path.write_text(json.dumps(expected))
    assert load_json(path) == expected


def test_map_coordinate_validation_rejects_invalid_wgs84_rows():
    source = pd.DataFrame({"lat": [10, 91, "bad"], "lon": [20, 30, 40]})
    result = valid_coordinate_rows(source, "lat", "lon")
    assert result[["lat", "lon"]].values.tolist() == [[10, 20]]


def test_incident_alignment_gate_rejects_jan_2_baseline_for_jan_7_incident():
    gate = SpatiotemporalValidationGate(max_temporal_gap_hours=24)
    valid, code, detail = gate.validate(
        pd.Timestamp("2024-01-07T10:00:00Z"),
        pd.Timestamp("2024-01-02T00:00:00Z"),
        pd.Timestamp("2024-01-02T23:59:59Z"),
        29.135,
        -89.3039,
        (27.5, 30.5),
        (-91.5, -88.0),
    )
    assert not valid
    assert code == "TEMPORAL_WINDOW_MISMATCH"
    assert detail["temporal_gap_hours"] > 24
