"""
neural_net.py – Keras MLP classifier for match prediction.
Classes: 0=HomeWin, 1=Draw, 2=AwayWin
"""
from typing import Optional

import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[NeuralNet] TensorFlow not available – NN model disabled.")

from sklearn.preprocessing import label_binarize
from models.base_model import BaseModel


class NeuralNetModel(BaseModel):
    name = "neural_net"

    def __init__(self, input_dim: int = 27):
        self.input_dim = input_dim
        self.model = None
        self.cv_score: Optional[float] = None
        self._trained = False

    def _build(self) -> "tf.keras.Model":
        inp = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(32)(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.2)(x)

        out = layers.Dense(3, activation="softmax")(x)
        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if not TF_AVAILABLE:
            print("[NeuralNet] Skipping – TensorFlow not installed.")
            return

        self.input_dim = X.shape[1]
        self.model = self._build()

        cb = [
            callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                     monitor="val_accuracy"),
            callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
        ]

        # Split 10% for validation
        split = int(0.9 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=64,
            callbacks=cb,
            verbose=0,
        )
        best_val_acc = max(history.history.get("val_accuracy", [0]))
        self.cv_score = round(best_val_acc, 4)
        print(f"[NeuralNet] Best val accuracy: {self.cv_score:.3f}")
        self._trained = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._trained or self.model is None:
            # Return uniform distribution if not trained
            n = X.shape[0]
            return np.full((n, 3), 1/3)
        return self.model.predict(X, verbose=0)
