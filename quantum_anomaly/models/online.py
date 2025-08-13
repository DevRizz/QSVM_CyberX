from typing import Optional, Tuple
import numpy as np
from river import anomaly

class StreamingBaseline:
    """
    Unsupervised streaming anomaly detector with River HalfSpaceTrees.
    Produces an anomaly score; higher means more anomalous.
    """
    def __init__(self):
        self.model = anomaly.HalfSpaceTrees(seed=42)

    def score_one(self, x: np.ndarray) -> float:
        # River expects a dict of features
        d = {f"x{i}": float(v) for i, v in enumerate(x)}
        return float(self.model.score_one(d))

    def learn_one(self, x: np.ndarray):
        d = {f"x{i}": float(v) for i, v in enumerate(x)}
        self.model = self.model.learn_one(d)
