"""Shared prototype settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
IMAGE_SIZE = 256
CLASS_NAMES = ("sea", "oil", "lookalike", "ship", "land")
WIND_FACTOR = 0.03
BACK_DRIFT_HOURS = (2, 4, 6, 12)
ORIGIN_BUFFER_KM = 5.0
SCORE_WEIGHTS = {"proximity": .40, "loitering": .25, "ais_gap": .20, "vessel_type": .15}
LOITER_SPEED_KNOTS = 2.0
AIS_LOOKBACK_HOURS = 12
TRAINING = {"epochs": 30, "lr": 3e-4, "patience": 8, "batch_size": 8}
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RUN_DIR = ROOT / "runs" / "exp1"
