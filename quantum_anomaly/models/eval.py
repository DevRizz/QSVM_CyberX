from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

def train_test_metrics(
    X: np.ndarray,
    y: np.ndarray,
    q_model,                 # must have fit(), predict(), predict_proba()
    c_model,                 # same interface
    test_size: float = 0.3,
    seed: int = 42
) -> Dict[str, Any]:
    # stratify only if both classes present
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)

    # Train
    q_model.fit(X_train, y_train, seed=seed) if hasattr(q_model, "fit") else q_model.fit(X_train, y_train)
    c_model.fit(X_train, y_train)

    # Predict
    yq = q_model.predict(X_test)
    pq = q_model.predict_proba(X_test)[:, 1]
    yc = c_model.predict(X_test)
    pc = c_model.predict_proba(X_test)[:, 1]

    # Metrics
    def m(y_true, y_pred, p):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, p)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_true, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_true, p)
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, p)
        return {
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc),
            "confusion": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()},
            "n_test": int(len(y_true)),
        }

    return {
        "quantum": m(y_test, yq, pq),
        "classical": m(y_test, yc, pc),
        "n_train": int(len(y_train)),
    }
