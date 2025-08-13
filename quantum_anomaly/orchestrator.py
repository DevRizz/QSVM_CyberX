import os
import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from quantum_anomaly.features.engine import extract_features_from_packets
from quantum_anomaly.models.online import StreamingBaseline
from quantum_anomaly.models.quantum_svm import QuantumKernelSVC
from quantum_anomaly.online.buffer import SlidingBuffer
from quantum_anomaly.storage.db import insert_event
from quantum_anomaly.security.qkd import bb84_simulate_key, derive_session_key
from quantum_anomaly.security.secure_channel import SecureChannel
from quantum_anomaly.utils.logging_setup import setup_logger

logger = setup_logger("orchestrator")

class Orchestrator:
    def __init__(self):
        self.baseline = StreamingBaseline()
        self.buffer = SlidingBuffer(maxlen=int(os.getenv("BUFFER_MAX", "512")))
        self.qsvc = QuantumKernelSVC(n_landmarks=int(os.getenv("QK_LANDMARKS", "64")))
        self.feature_cols: Optional[List[str]] = None
        self.secure: Optional[SecureChannel] = None
        self._init_qkd()

    def _init_qkd(self):
        seed = os.getenv("QKD_SEED")
        seed = int(seed) if seed else None
        raw, err, n = bb84_simulate_key(length=2048, seed=seed, error_rate=0.01)
        logger.info(f"QKD: sifted={n}, observed_error={err:.3f}")
        if err < 0.11:  # BB84 QBER threshold guideline
            key = derive_session_key(raw, out_len=32)
            self.secure = SecureChannel(key)
            logger.info("QKD session key established (AES-256-GCM).")
        else:
            logger.warning("QKD aborted due to high error; falling back to local (unencrypted) mode.")
            self.secure = None

    def _select_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        # Use scaled features assembled in engine.py
        cols = [c for c in df.columns if c.endswith("_scaled")]
        if not cols:
            return np.empty((0, ))
        self.feature_cols = cols
        X = df[cols].fillna(0.0).to_numpy(dtype=float)
        return X

    def process_packets(self, packets) -> Dict[str, Any]:
        df = extract_features_from_packets(packets)
        X = self._select_feature_vector(df)
        scores = []
        preds = []
        for x in X:
            s = self.baseline.score_one(x)
            scores.append(s)
            self.baseline.learn_one(x)
            self.buffer.add(x, None)
        scores = np.array(scores, dtype=float)

        # If QSVC is trained, also produce predictions
        if self.qsvc.fitted and len(X):
            proba = self.qsvc.predict_proba(X)
            preds = (proba[:, 1] > 0.5).astype(int).tolist()
        else:
            preds = [int(s > 0.8) for s in scores]  # heuristic threshold

        # Log first row as example
        if len(df):
            event = {
                "src_ip": df.iloc[0].get("src_ip"),
                "dst_ip": df.iloc[0].get("dst_ip"),
                "protocol": df.iloc[0].get("protocol"),
                "features_json": json.dumps(df.iloc[0].to_dict()),
                "anomaly_score": float(scores[0]),
                "label": None,
                "note": "live" if "timestamp" in df.columns else "pcap",
            }
            if self.secure:
                # Encrypt features_json as a demo of pipeline security
                nonce, ct = self.secure.encrypt(event["features_json"].encode("utf-8"))
                event["features_json"] = json.dumps({"nonce": nonce.hex(), "ct": ct.hex()})
            insert_event(event)

        return {
            "df": df,
            "scores": scores.tolist(),
            "preds": preds,
        }

    def update_with_label(self, indices: List[int], label: int, df: pd.DataFrame):
        """
        User labels selected rows as anomaly(1)/normal(0). Update buffer and retrain QSVC if enough data.
        """
        X_all = self._select_feature_vector(df)
        for i in indices:
            if 0 <= i < len(X_all):
                self.buffer.add(X_all[i], label)

        # Retrain QSVC if enough labeled samples exist
        min_train = int(os.getenv("BUFFER_MIN_TRAIN", "64"))
        X_lab, y_lab = self.buffer.labeled()
        if len(X_lab) >= min_train:
            logger.info(f"Retraining Quantum SVC on {len(X_lab)} labeled samples")
            self.qsvc.fit(X_lab, y_lab, seed=42)

    def predict_from_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        X = self._select_feature_vector(df)
        scores = []
        for x in X:
            s = self.baseline.score_one(x)
            scores.append(s)
        scores = np.array(scores)
        preds = [int(s > 0.8) for s in scores]
        if self.qsvc.fitted and len(X):
            proba = self.qsvc.predict_proba(X)
            preds = (proba[:, 1] > 0.5).astype(int).tolist()
        return {"scores": scores.tolist(), "preds": preds}
