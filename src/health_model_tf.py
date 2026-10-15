import numpy as np
import joblib
import tensorflow as tf
from typing import Dict

MODEL_PATH = "src/models/health_risk_tf_model.h5"
META_PATH = "src/models/tf_metadata.pkl"

_label_map = {0: "Low", 1: "Moderate", 2: "High"}

class HealthRiskModelTF:
    def __init__(self):
        # Load trained TensorFlow model
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Load metadata (scaler + column order)
        meta = joblib.load(META_PATH)
        self.scaler = meta["scaler"]
        self.feature_cols = meta["feature_cols"]

    def preprocess(self, user_input: Dict) -> np.ndarray:
        # Build feature vector in correct order
        x = np.array([user_input.get(col, 0) for col in self.feature_cols], dtype="float32")
        x = x.reshape(1, -1)
        x = self.scaler.transform(x)
        return x

    def predict(self, user_input: Dict):
        x = self.preprocess(user_input)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))

        return {
            "risk_label": _label_map[idx],
            "risk_index": idx,
            "risk_proba": {
                "Low": float(probs[0]),
                "Moderate": float(probs[1]),
                "High": float(probs[2]),
            },
        }


# Singleton instance
_model_tf = None

def get_health_model_tf():
    global _model_tf
    if _model_tf is None:
        _model_tf = HealthRiskModelTF()
    return _model_tf
