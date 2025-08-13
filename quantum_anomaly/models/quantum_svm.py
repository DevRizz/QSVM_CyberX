from typing import Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from quantum_anomaly.quantum.kernel import NystromQuantumKernel

class QuantumKernelSVC:
    """
    Binary SVC trained with features transformed via Nystrom quantum kernel mapping.
    Re-train on labeled sliding window as feedback arrives.
    """
    def __init__(self, n_landmarks: int = 64):
        self.nystrom = NystromQuantumKernel(n_landmarks=n_landmarks)
        self.clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svc", SVC(kernel="linear", probability=True, class_weight="balanced"))
        ])
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, seed: Optional[int] = None):
        if len(X) == 0:
            return
        self.nystrom.fit_landmarks(X, seed=seed)
        Phi = self.nystrom.transform(X)
        self.clf.fit(Phi, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or len(X) == 0:
            return np.zeros((len(X),), dtype=int)
        Phi = self.nystrom.transform(X)
        return self.clf.predict(Phi)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or len(X) == 0:
            return np.full((len(X), 2), fill_value=0.5)
        Phi = self.nystrom.transform(X)
        return self.clf.predict_proba(Phi)
