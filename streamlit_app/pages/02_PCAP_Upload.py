# import io
# import json
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from quantum_anomaly.capture.pcap_loader import load_pcap
# from quantum_anomaly.orchestrator import Orchestrator

# st.set_page_config(page_title="PCAP Upload", layout="wide")
# st.title("PCAP Upload")

# if "orchestrator" not in st.session_state:
#     st.session_state["orchestrator"] = Orchestrator()
# orch: Orchestrator = st.session_state["orchestrator"]

# file = st.file_uploader("Upload PCAP", type=["pcap", "pcapng"])
# if file:
#     st.success("PCAP uploaded.")
#     packets = load_pcap(file)
#     st.write(f"Packets loaded: {len(packets)}")

#     result = orch.process_packets(packets)
#     df = result["df"]
#     scores = np.array(result["scores"])
#     preds = np.array(result["preds"])

#     st.dataframe(df.head(50), use_container_width=True)

#     if len(scores):
#         fig = px.histogram(scores, nbins=30, title="Anomaly Score Distribution")
#         st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Feedback")
#     with st.form("label_form"):
#         st.caption("Label selected indices (comma-separated, 0-based)")
#         idx_text = st.text_input("Indices", value="0,1,2")
#         label = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
#         submitted = st.form_submit_button("Update model")
#         if submitted:
#             try:
#                 idxs = [int(x.strip()) for x in idx_text.split(",") if x.strip().isdigit()]
#                 lab = 0 if label.startswith("Normal") else 1
#                 orch.update_with_label(indices=idxs, label=lab, df=df)
#                 st.success(f"Labeled {len(idxs)} rows as {lab}. Quantum SVC trained: {orch.qsvc.fitted}")
#             except Exception as e:
#                 st.error(str(e))


import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st
import plotly.express as px
from quantum_anomaly.ui import theme
from quantum_anomaly.capture.pcap_loader import load_pcap
from quantum_anomaly.orchestrator import Orchestrator

st.set_page_config(page_title="PCAP Upload", layout="wide", page_icon="📦")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Offline PCAP Analysis",
    "Upload traffic captures to extract features, score anomalies, and provide labels.",
    lottie_url="https://lottie.host/c2f5be4f-0b6d-4b43-96c3-3b6f2fc0d1ef/0w2eK2ZC7E.json"
)

if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()
orch: Orchestrator = st.session_state["orchestrator"]

file = st.file_uploader("Upload .pcap or .pcapng", type=["pcap", "pcapng"])
if file:
    st.success("PCAP uploaded.")
    packets = load_pcap(file)
    st.write(f"Packets loaded: {len(packets)}")

    result = orch.process_packets(packets)
    df = result["df"]; scores = np.array(result["scores"]); preds = np.array(result["preds"])

    st.dataframe(df.head(50), use_container_width=True)

    if len(scores):
        fig = px.histogram(scores, nbins=30, title="Anomaly Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feedback")
    with st.form("label_form"):
        st.caption("Label selected indices (comma-separated, 0-based)")
        idx_text = st.text_input("Indices", value="0,1,2")
        label = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
        submitted = st.form_submit_button("Update model")
        if submitted:
            try:
                idxs = [int(x.strip()) for x in idx_text.split(",") if x.strip().isdigit()]
                lab = 0 if label.startswith("Normal") else 1
                orch.update_with_label(indices=idxs, label=lab, df=df)
                st.success(f"Labeled {len(idxs)} rows as {lab}. Quantum SVC trained: {orch.qsvc.fitted}")
                if orch.qsvc.fitted: st.balloons()
            except Exception as e:
                st.error(str(e))