import os
import sqlite3
from contextlib import contextmanager
from typing import Optional, Dict, Any
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "./anomaly_events.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            src_ip TEXT,
            dst_ip TEXT,
            protocol TEXT,
            features_json TEXT,
            anomaly_score REAL,
            label INTEGER,
            note TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            description TEXT,
            details_json TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS secrets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            key_label TEXT UNIQUE,
            cipher_params_json TEXT
        )
        """)
        con.commit()

@contextmanager
def get_conn():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
    finally:
        con.close()

def insert_event(event: Dict[str, Any]):
    event = {**event}
    event.setdefault("ts", datetime.utcnow().isoformat())
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("""
        INSERT INTO events (ts, src_ip, dst_ip, protocol, features_json, anomaly_score, label, note)
        VALUES (:ts, :src_ip, :dst_ip, :protocol, :features_json, :anomaly_score, :label, :note)
        """, event)
        con.commit()

def insert_anomaly(description: str, details_json: str):
    with get_conn() as con:
        cur = con.cursor()
        cur.execute("""
        INSERT INTO anomalies (ts, description, details_json)
        VALUES (?, ?, ?)
        """, (datetime.utcnow().isoformat(), description, details_json))
        con.commit()
