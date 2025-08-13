# # # import time
# # # import json
# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.express as px

# # # from quantum_anomaly.capture.live_sniffer import LiveSniffer
# # # from quantum_anomaly.orchestrator import Orchestrator

# # # st.set_page_config(page_title="Live Capture", layout="wide")
# # # st.title("Live Capture")

# # # if "orchestrator" not in st.session_state:
# # #     st.session_state["orchestrator"] = Orchestrator()

# # # if "sniffer" not in st.session_state:
# # #     st.session_state["sniffer"] = None
# # # if "packets_buffer" not in st.session_state:
# # #     st.session_state["packets_buffer"] = []

# # # orch: Orchestrator = st.session_state["orchestrator"]

# # # with st.sidebar:
# # #     st.subheader("Sniffer Controls")
# # #     if st.session_state["sniffer"] is None:
# # #         ifaces = LiveSniffer.list_interfaces()
# # #         iface = st.selectbox("Network interface", options=list(ifaces.keys()) or [""], index=0)
# # #         bpf = st.text_input("BPF filter (optional)", value="tcp or udp")
# # #         start = st.button("Start Sniffing", type="primary")
# # #         if start:
# # #             sniffer = LiveSniffer(iface=iface if iface else None, bpf_filter=bpf or None)

# # #             def on_packet(pkt):
# # #                 st.session_state["packets_buffer"].append(pkt)

# # #             st.session_state["sniffer"] = sniffer
# # #             sniffer.start(on_packet=on_packet)
# # #             st.success("Sniffer started.")
# # #     else:
# # #         if st.button("Stop Sniffing", type="secondary"):
# # #             st.session_state["sniffer"].stop()
# # #             st.session_state["sniffer"] = None
# # #             st.success("Sniffer stopped.")

# # # col1, col2 = st.columns([2, 1])

# # # with col1:
# # #     st.subheader("Stream")
# # #     run = st.toggle("Run streaming analysis", value=False)
# # #     placeholder = st.empty()
# # #     chart_ph = st.empty()
# # #     feats_ph = st.empty()

# # # with col2:
# # #     st.subheader("Actions")
# # #     lbl_col = st.container(border=True)
# # #     with lbl_col:
# # #         st.caption("Label last N rows to improve the Quantum SVC")
# # #         n_last = st.number_input("N rows", min_value=1, max_value=512, value=32, step=1)
# # #         lab = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
# # #         if st.button("Apply Label"):
# # #             if "last_df" in st.session_state and len(st.session_state["last_df"]) > 0:
# # #                 df = st.session_state["last_df"]
# # #                 idxs = list(range(max(0, len(df) - n_last), len(df)))
# # #                 label = 0 if lab.startswith("Normal") else 1
# # #                 orch.update_with_label(indices=idxs, label=label, df=df)
# # #                 st.success(f"Labeled {len(idxs)} rows as {label}")

# # # if run:
# # #     while True:
# # #         pkts = st.session_state["packets_buffer"]
# # #         st.session_state["packets_buffer"] = []
# # #         if pkts:
# # #             result = orch.process_packets(pkts)
# # #             df = result["df"]
# # #             scores = result["scores"]
# # #             preds = result["preds"]
# # #             st.session_state["last_df"] = df

# # #             # Display
# # #             with placeholder.container():
# # #                 st.write(f"Processed packets: {len(df)}")
# # #                 st.dataframe(df.tail(15), use_container_width=True)

# # #             if len(df):
# # #                 x = list(range(len(scores)))
# # #                 fig = px.line(x=x, y=scores, labels={"x": "Sample", "y": "Anomaly Score"}, title="Streaming Anomaly Score")
# # #                 chart_ph.plotly_chart(fig, use_container_width=True)

# # #                 st.metric("Quantum SVC Trained", value=str(orch.qsvc.fitted))
# # #         else:
# # #             time.sleep(0.5)
# # #         # Break the loop when user toggles off
# # #         if not st.session_state.get("_rerun_guard", False) and not run:
# # #             break
# # #         # yield control to Streamlit
# # #         st.experimental_rerun()
# # # else:
# # #     st.info("Toggle 'Run streaming analysis' to process incoming packets.")

# # import time
# # import json
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.express as px

# # from quantum_anomaly.capture.live_sniffer import LiveSniffer
# # from quantum_anomaly.orchestrator import Orchestrator

# # st.set_page_config(page_title="Live Capture", layout="wide")
# # st.title("Live Capture")

# # if "orchestrator" not in st.session_state:
# #   st.session_state["orchestrator"] = Orchestrator()

# # if "sniffer" not in st.session_state:
# #   st.session_state["sniffer"] = None
# # if "packets_buffer" not in st.session_state:
# #   st.session_state["packets_buffer"] = []

# # orch: Orchestrator = st.session_state["orchestrator"]

# # with st.sidebar:
# #   st.subheader("Sniffer Controls")
# #   if st.session_state["sniffer"] is None:
# #       ifaces = LiveSniffer.list_interfaces()
# #       iface = st.selectbox("Network interface", options=list(ifaces.keys()) or [""], index=0)
# #       bpf = st.text_input("BPF filter (optional)", value="tcp or udp")
# #       start = st.button("Start Sniffing", type="primary")
# #       if start:
# #           sniffer = LiveSniffer(iface=iface if iface else None, bpf_filter=bpf or None)

# #           def on_packet(pkt):
# #               st.session_state["packets_buffer"].append(pkt)

# #           st.session_state["sniffer"] = sniffer
# #           sniffer.start(on_packet=on_packet)
# #           st.success("Sniffer started.")
# #   else:
# #       if st.button("Stop Sniffing", type="secondary"):
# #           st.session_state["sniffer"].stop()
# #           st.session_state["sniffer"] = None
# #           st.success("Sniffer stopped.")

# # col1, col2 = st.columns([2, 1])

# # with col1:
# #   st.subheader("Stream")
# #   run = st.toggle("Run streaming analysis", value=False)
# #   placeholder = st.empty()
# #   chart_ph = st.empty()
# #   feats_ph = st.empty()

# # with col2:
# #   st.subheader("Actions")
# #   lbl_col = st.container(border=True)
# #   with lbl_col:
# #       st.caption("Label last N rows to improve the Quantum SVC")
# #       n_last = st.number_input("N rows", min_value=1, max_value=512, value=32, step=1)
# #       lab = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
# #       if st.button("Apply Label"):
# #           if "last_df" in st.session_state and len(st.session_state["last_df"]) > 0:
# #               df = st.session_state["last_df"]
# #               idxs = list(range(max(0, len(df) - n_last), len(df)))
# #               label = 0 if lab.startswith("Normal") else 1
# #               orch.update_with_label(indices=idxs, label=label, df=df)
# #               st.success(f"Labeled {len(idxs)} rows as {label}")

# # if run:
# #   while True:
# #       pkts = st.session_state["packets_buffer"]
# #       st.session_state["packets_buffer"] = []
# #       if pkts:
# #           result = orch.process_packets(pkts)
# #           df = result["df"]
# #           scores = result["scores"]
# #           preds = result["preds"]
# #           st.session_state["last_df"] = df

# #           # Display
# #           with placeholder.container():
# #               st.write(f"Processed packets: {len(df)}")
# #               st.dataframe(df.tail(15), use_container_width=True)

# #           if len(df):
# #               x = list(range(len(scores)))
# #               fig = px.line(x=x, y=scores, labels={"x": "Sample", "y": "Anomaly Score"}, title="Streaming Anomaly Score")
# #               chart_ph.plotly_chart(fig, use_container_width=True)

# #               st.metric("Quantum SVC Trained", value=str(orch.qsvc.fitted))
# #       else:
# #           time.sleep(0.5)
# #       # Break the loop when user toggles off
# #       if not st.session_state.get("_rerun_guard", False) and not run:
# #           break
# #       # yield control to Streamlit
# #       st.rerun()
# # else:
# #   st.info("Toggle 'Run streaming analysis' to process incoming packets.")

# import time
# import json
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from quantum_anomaly.capture.live_sniffer import LiveSniffer
# from quantum_anomaly.orchestrator import Orchestrator

# # CSS injection for new styles
# try:
#     import os
#     with open(os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css"), "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# except Exception:
#     pass

# st.set_page_config(page_title="Live Capture", layout="wide")

# # Gradient banner
# st.markdown('<div class="banner">Stream live traffic, compute features, score anomalies, and provide labels to train the Quantum SVC.</div>', unsafe_allow_html=True)

# if "orchestrator" not in st.session_state:
#   st.session_state["orchestrator"] = Orchestrator()

# if "sniffer" not in st.session_state:
#   st.session_state["sniffer"] = None
# if "packets_buffer" not in st.session_state:
#   st.session_state["packets_buffer"] = []

# orch: Orchestrator = st.session_state["orchestrator"]

# with st.sidebar:
#   st.subheader("Sniffer Controls")
#   if st.session_state["sniffer"] is None:
#       ifaces = LiveSniffer.list_interfaces()
#       iface = st.selectbox("Network interface", options=list(ifaces.keys()) or [""], index=0)
#       bpf = st.text_input("BPF filter (optional)", value="tcp or udp")
#       start = st.button("Start Sniffing", type="primary")
#       if start:
#           sniffer = LiveSniffer(iface=iface if iface else None, bpf_filter=bpf or None)

#           def on_packet(pkt):
#               st.session_state["packets_buffer"].append(pkt)

#           st.session_state["sniffer"] = sniffer
#           sniffer.start(on_packet=on_packet)
#           st.success("Sniffer started.")
#   else:
#       if st.button("Stop Sniffing", type="secondary"):
#           st.session_state["sniffer"].stop()
#           st.session_state["sniffer"] = None
#           st.success("Sniffer stopped.")

# col1, col2 = st.columns([2, 1])

# with col1:
#   st.subheader("Stream")
#   run = st.toggle("Run streaming analysis", value=False)
#   placeholder = st.empty()
#   chart_ph = st.empty()
#   feats_ph = st.empty()

#   # Metric row showing buffer size and QSVC trained state
#   m1, m2 = st.columns(2)
#   m1.metric("Buffer size", value=str(len(st.session_state.get("packets_buffer", []))))
#   m2.metric("Quantum SVC Trained", value=str(orch.qsvc.fitted))

# with col2:
#   st.subheader("Actions")
#   lbl_col = st.container(border=True)
#   with lbl_col:
#       st.caption("Label last N rows to improve the Quantum SVC")
#       n_last = st.number_input("N rows", min_value=1, max_value=512, value=32, step=1)
#       lab = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
#       if st.button("Apply Label"):
#           if "last_df" in st.session_state and len(st.session_state["last_df"]) > 0:
#               df = st.session_state["last_df"]
#               idxs = list(range(max(0, len(df) - n_last), len(df)))
#               label = 0 if lab.startswith("Normal") else 1
#               orch.update_with_label(indices=idxs, label=label, df=df)
#               st.success(f"Labeled {len(idxs)} rows as {label}")

# if run:
#   while True:
#       pkts = st.session_state["packets_buffer"]
#       st.session_state["packets_buffer"] = []
#       if pkts:
#           result = orch.process_packets(pkts)
#           df = result["df"]
#           scores = result["scores"]
#           preds = result["preds"]
#           st.session_state["last_df"] = df

#           # Display
#           with placeholder.container():
#               st.write(f"Processed packets: {len(df)}")
#               st.dataframe(df.tail(15), use_container_width=True)

#           if len(df):
#               x = list(range(len(scores)))
#               fig = px.line(x=x, y=scores, labels={"x": "Sample", "y": "Anomaly Score"}, title="Streaming Anomaly Score")
#               chart_ph.plotly_chart(fig, use_container_width=True)
#       else:
#           time.sleep(0.5)
#       # Break the loop when user toggles off
#       if not st.session_state.get("_rerun_guard", False) and not run:
#           break
#       # yield control to Streamlit
#       st.rerun()
# else:
#   st.info("Toggle 'Run streaming analysis' to process incoming packets.")

import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
from quantum_anomaly.ui import theme
from quantum_anomaly.capture.live_sniffer import LiveSniffer
from quantum_anomaly.orchestrator import Orchestrator

st.set_page_config(page_title="Live Capture", layout="wide", page_icon="🛰️")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Live Capture & Streaming Analysis",
    "Sniff live packets, compute features, track anomaly scores, and label to train QSVM.",
    lottie_url="https://lottie.host/0dc9b7b0-1b14-49a3-86d3-4d6b45d8510f/Pzbj1S2gS9.json"
)

if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()
if "sniffer" not in st.session_state:
    st.session_state["sniffer"] = None
if "packets_buffer" not in st.session_state:
    st.session_state["packets_buffer"] = []
if "celebrated" not in st.session_state:
    st.session_state["celebrated"] = False

orch: Orchestrator = st.session_state["orchestrator"]

with st.sidebar:
    st.subheader("Sniffer Controls")
    if st.session_state["sniffer"] is None:
        ifaces = LiveSniffer.list_interfaces()
        iface = st.selectbox("Network interface", options=list(ifaces.keys()) or [""], index=0)
        bpf = st.text_input("BPF filter (optional)", value="tcp or udp")
        if st.button("Start Sniffing", type="primary"):
            sniffer = LiveSniffer(iface=iface if iface else None, bpf_filter=bpf or None)
            def on_packet(pkt):
                st.session_state["packets_buffer"].append(pkt)
            st.session_state["sniffer"] = sniffer
            sniffer.start(on_packet=on_packet)
            st.success("Sniffer started.")
    else:
        if st.button("Stop Sniffing", type="secondary"):
            st.session_state["sniffer"].stop()
            st.session_state["sniffer"] = None
            st.success("Sniffer stopped.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Streaming")
    run = st.toggle("Run streaming analysis", value=False)
    placeholder = st.empty()
    chart_ph = st.empty()
    m1, m2, m3 = st.columns(3)
    m1.metric("Buffer (packets)", value=str(len(st.session_state.get("packets_buffer", []))))
    m2.metric("QSVM Trained", value=str(orch.qsvc.fitted))
    m3.metric("QKD Secure", value="Yes" if orch.secure else "No")

with col2:
    st.subheader("Actions")
    with st.container():
        st.caption("Label last N rows to improve the Quantum SVC")
        n_last = st.number_input("N rows", min_value=1, max_value=512, value=32, step=1)
        lab = st.selectbox("Label", options=["Normal (0)", "Anomaly (1)"], index=0)
        if st.button("Apply Label"):
            if "last_df" in st.session_state and len(st.session_state["last_df"]) > 0:
                df = st.session_state["last_df"]
                idxs = list(range(max(0, len(df) - n_last), len(df)))
                label = 0 if lab.startswith("Normal") else 1
                orch.update_with_label(indices=idxs, label=label, df=df)
                st.success(f"Labeled {len(idxs)} rows as {label}")

if run:
    while True:
        pkts = st.session_state["packets_buffer"]
        st.session_state["packets_buffer"] = []
        if pkts:
            result = orch.process_packets(pkts)
            df = result["df"]
            scores = result["scores"]
            preds = result["preds"]
            st.session_state["last_df"] = df

            with placeholder.container():
                st.write(f"Processed packets: {len(df)}")
                st.dataframe(df.tail(15), use_container_width=True)

            if len(df):
                x = list(range(len(scores)))
                fig = px.line(x=x, y=scores, labels={"x": "Sample", "y": "Anomaly Score"}, title="Streaming Anomaly Score")
                chart_ph.plotly_chart(fig, use_container_width=True)

            if orch.qsvc.fitted and not st.session_state["celebrated"]:
                st.balloons()
                st.session_state["celebrated"] = True
        else:
            time.sleep(0.5)
        if not run:
            break
        st.rerun()
else:
    st.info("Toggle 'Run streaming analysis' to process incoming packets.")