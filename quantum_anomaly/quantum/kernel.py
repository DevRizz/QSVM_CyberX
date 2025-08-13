from typing import Optional, Tuple
import os
import numpy as np
import pennylane as qml
from pennylane.kernels import square_kernel_matrix

# Config
QK_WIRES = int(os.getenv("QK_WIRES", "4"))
QK_EMBED_SCALE = float(os.getenv("QK_EMBED_SCALE", "1.0"))

dev = qml.device("default.qubit", wires=QK_WIRES, shots=None)

@qml.qnode(dev)
def embedding_circuit(x):
    # Truncate or pad x to number of wires
    w = QK_WIRES
    if len(x) < w:
        xx = np.pad(x, (0, w - len(x)), mode="constant")
    else:
        xx = np.array(x[:w])
    # Scale features (angle embedding is periodic)
    xx = QK_EMBED_SCALE * xx
    qml.AngleEmbedding(xx, wires=range(w))
    qml.BasicEntanglerLayers(weights=np.ones((1, w)))
    return qml.state()

def quantum_kernel_matrix(X: np.ndarray) -> np.ndarray:
    """
    Full square kernel matrix using state fidelity via PennyLane.
    O(n^2) — use Nyström for online settings.
    """
    return square_kernel_matrix(X, embedding_circuit)

class NystromQuantumKernel:
    """
    Nyström approximation for quantum kernel K ≈ C W^+ C^T
    - Landmarks L (subset of X)
    - W = K(L, L)
    - For new x: k_xL = K(x, L), then phi_x = k_xL @ W^(-1/2)
    - Use phi as feature map into linear classifier / SVC with linear kernel
    """
    def __init__(self, n_landmarks: int = 64, reg: float = 1e-6):
        self.n_landmarks = n_landmarks
        self.reg = reg
        self.L: Optional[np.ndarray] = None
        self.W_eigvec: Optional[np.ndarray] = None
        self.W_eigval_inv_sqrt: Optional[np.ndarray] = None

    def fit_landmarks(self, X: np.ndarray, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        if len(X) <= self.n_landmarks:
            self.L = np.array(X, dtype=float)
        else:
            idx = rng.choice(len(X), size=self.n_landmarks, replace=False)
            self.L = np.array(X[idx], dtype=float)
        # Compute W
        W = quantum_kernel_matrix(self.L)
        # Regularize
        evals, evecs = np.linalg.eigh(W + self.reg * np.eye(W.shape[0]))
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(evals, self.reg)))
        self.W_eigvec = evecs
        self.W_eigval_inv_sqrt = inv_sqrt

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.L is not None and self.W_eigvec is not None and self.W_eigval_inv_sqrt is not None
        # Compute C = K(X, L)
        # For efficiency, batch compute rows
        feats = []
        for x in X:
            C_row = []
            for l in self.L:
                k = square_kernel_matrix(np.stack([x, l]), embedding_circuit)[0, 1]
                C_row.append(k)
            C_row = np.array(C_row, dtype=float)
            # phi_x = C_row * evecs * inv_sqrt
            phi = C_row @ self.W_eigvec @ self.W_eigval_inv_sqrt
            feats.append(phi)
        return np.array(feats)
