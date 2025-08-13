import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from quantum_anomaly.ui import theme

st.set_page_config(page_title="Cyber Ops", layout="wide", page_icon="🧰")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Cyber Security Ops",
    "A dedicated space to emphasize the non-quantum cybersecurity aspects of the project.",
    lottie_url="https://lottie.host/8362d7dd-6b64-4d35-931c-814d6e3c2e9d/Wx8kGqHj2x.json"
)

c1, c2, c3 = st.columns(3)
with c1: st.markdown("""<div class="card"><h3>Threat Modeling</h3><p class="small">MITM on routing, port scans, DDoS spikes. We set thresholds, visualize distributions, and alert on deviations.</p></div>""", unsafe_allow_html=True)
with c2: st.markdown("""<div class="card"><h3>Telemetry & Logging</h3><p class="small">SQLite logs for events and anomalies; optional encryption of feature payloads with QKD-derived AES-GCM.</p></div>""", unsafe_allow_html=True)
with c3: st.markdown("""<div class="card"><h3>Operator Workflow</h3><p class="small">Live Capture to observe, PCAP Upload to replay incidents, Dashboard to pivot, QSVM to learn from feedback.</p></div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
st.markdown("""
- Cyber page showcases the classical ops story: baselines, heuristics, dashboards, and secured logging.
- Quantum page highlights the research component: kernel embeddings, comparison with classical SVMs, and online improvements.
""")
