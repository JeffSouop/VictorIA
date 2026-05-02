"""
base_model.py – Abstract base class for all prediction models.
"""
from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "cache"


class BaseModel(ABC):
    """All models must implement this interface."""

    name: str = "base"

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix (n_samples, 3) for [HomeWin, Draw, AwayWin]."""
        ...

    def save(self) -> None:
        path = MODEL_DIR / f"{self.name}.pkl"
        joblib.dump(self, path)
        print(f"[{self.name}] Saved to {path}")

    @classmethod
    def load(cls, name: str) -> "BaseModel":
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        return joblib.load(path)

    def is_cached(self) -> bool:
        return (MODEL_DIR / f"{self.name}.pkl").exists()
