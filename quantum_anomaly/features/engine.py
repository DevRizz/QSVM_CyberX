from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import math
import time
from scapy.all import Packet, IP, TCP, UDP

def _port(pkt) -> Tuple[int, int]:
    sport = dport = None
    if TCP in pkt:
        sport = int(pkt[TCP].sport)
        dport = int(pkt[TCP].dport)
    elif UDP in pkt:
        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)
    return sport or -1, dport or -1

def _flags(pkt) -> Dict[str, int]:
    flags = dict(SYN=0, ACK=0, FIN=0, RST=0, PSH=0, URG=0)
    if TCP in pkt:
        f = pkt[TCP].flags
        flags["SYN"] = int(bool(f & 0x02))
        flags["ACK"] = int(bool(f & 0x10))
        flags["FIN"] = int(bool(f & 0x01))
        flags["RST"] = int(bool(f & 0x04))
        flags["PSH"] = int(bool(f & 0x08))
        flags["URG"] = int(bool(f & 0x20))
    return flags

def extract_features_from_packets(packets: List[Packet]) -> pd.DataFrame:
    rows = []
    last_ts = None
    for pkt in packets:
        try:
            ts = float(pkt.time)
        except Exception:
            ts = time.time()
        if last_ts is None:
            delta = 0.0
        else:
            delta = max(0.0, ts - last_ts)
        last_ts = ts

        length = int(len(bytes(pkt)))
        src_ip = pkt[IP].src if IP in pkt else "0.0.0.0"
        dst_ip = pkt[IP].dst if IP in pkt else "0.0.0.0"
        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
        sport, dport = _port(pkt)
        flags = _flags(pkt)

        rows.append({
            "timestamp": ts,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": proto,
            "src_port": sport,
            "dst_port": dport,
            "length": length,
            "delta_t": delta,
            **{f"flag_{k.lower()}": v for k, v in flags.items()},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Aggregate to rolling windows or return per-packet features with additional context
    # For simplicity, compute per-packet derived features:
    df["bytes_cum_src"] = df.groupby("src_ip")["length"].cumsum()
    df["bytes_cum_dst"] = df.groupby("dst_ip")["length"].cumsum()

    # Exponential moving averages for inter-arrival
    df["ema_delta_t"] = df["delta_t"].ewm(alpha=0.3).mean()

    # Port entropy (approx) using rolling window of last 50 dst ports
    dst_ports = deque(maxlen=50)
    entropies = []
    for p in df["dst_port"].values:
        dst_ports.append(p)
        counts = pd.Series(dst_ports).value_counts()
        probs = counts / counts.sum()
        ent = float(-(probs * np.log2(probs)).sum())
        entropies.append(ent)
    df["dst_port_entropy"] = entropies

    # Normalize main numeric features (simple scaling; quantum kernels are scale-sensitive)
    features = [
        "length", "ema_delta_t",
        "flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg",
        "dst_port_entropy"
    ]
    X = df[features].fillna(0.0).to_numpy(dtype=float)
    # Robust scaling
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-6
    X_scaled = (X - med) / mad
    out = df.copy()
    for i, f in enumerate(features):
        out[f"{f}_scaled"] = X_scaled[:, i]
    return out
