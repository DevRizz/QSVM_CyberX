from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ClassicalSVM:
    """
    Classical SVM with RBF kernel for baseline comparison.
    """
    def __init__(self):
        self.clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale"))
        ])
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0:
            return
        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or len(X) == 0:
            return np.zeros((len(X),), dtype=int)
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or len(X) == 0:
            return np.full((len(X), 2), 0.5, dtype=float)
        return self.clf.predict_proba(X)
