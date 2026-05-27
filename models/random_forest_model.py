"""
random_forest_model.py – Calibrated Random Forest classifier.
Classes: 0=HomeWin, 1=Draw, 2=AwayWin
"""
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models.base_model import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self):
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        )
        self.model = CalibratedClassifierCV(base_rf, method="isotonic", cv=5)
        self.cv_score: Optional[float] = None
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        base_rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            max_features="sqrt", random_state=42, n_jobs=1,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            base_rf, X, y, cv=cv, scoring="accuracy", n_jobs=1
        )
        self.cv_score = float(scores.mean())
        print(f"[RandomForest] CV Accuracy: {self.cv_score:.3f} ± {scores.std():.3f}")
        self.model.fit(X, y)
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained.")
        return self.model.predict_proba(X)
