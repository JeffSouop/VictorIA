"""
config.py – Central configuration for VictorIA.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ENSEMBLE_PATH = CACHE_DIR / "ensemble.pkl"
SCALER_PATH = CACHE_DIR / "scaler.pkl"
METRICS_PATH = CACHE_DIR / "metrics.json"

# Training
TRAINING_SAMPLES = 2000
TRAINING_TEST_SIZE = 0.2
TRAINING_RANDOM_STATE = 42

# Ensemble weights [XGBoost, RandomForest, NeuralNet]
ENSEMBLE_WEIGHTS = [0.55, 0.45, 0.0]  # NN disabled by default (no TensorFlow)

# Bump when logic changes to invalidate Streamlit cache
APP_VERSION = "1.3.0"

# Blend ML output with empirical stats when real API data is available
EMPIRICAL_BLEND_WEIGHT = 0.75

# Real historical training (football-data.org)
COMPETITION_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Ligue 1": "FL1",
    "Serie A": "SA",
    "Bundesliga": "BL1",
}
TOP5_COMPETITION_CODES = ["PL", "PD", "FL1", "SA", "BL1"]
REAL_DATASET_PATH = CACHE_DIR / "real_dataset.pkl"
REAL_DATASET_META_PATH = CACHE_DIR / "real_dataset_meta.json"
DEFAULT_TRAINING_SEASONS = 4  # number of past seasons to fetch
MIN_TEAM_MATCHES = 3
STATS_WINDOW = 10
API_REQUEST_DELAY_SEC = 6.5  # free tier ~10 req/min
