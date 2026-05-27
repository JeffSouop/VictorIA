"""
xgboost_model.py – XGBoost multi-class classifier for match prediction.
Classes: 0=HomeWin, 1=Draw, 2=AwayWin
"""
from typing import Optional

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.80,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=1,
        )
        self.cv_score: Optional[float] = None
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model, X, y, cv=cv, scoring="accuracy", n_jobs=1
        )
        self.cv_score = float(scores.mean())
        print(f"[XGBoost] CV Accuracy: {self.cv_score:.3f} ± {scores.std():.3f}")
        self.model.fit(X, y)
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_
