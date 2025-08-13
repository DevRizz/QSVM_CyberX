# # import os
# # import streamlit as st

# # st.set_page_config(page_title="Quantum Anomaly Detection (QKD-secured)", layout="wide")

# # st.title("Quantum Kernel Anomaly Detection for Network Traffic")
# # st.caption("Real-time features + Online learning + Quantum kernel SVC + QKD-secured pipeline")

# # st.sidebar.header("Navigation")
# # st.sidebar.markdown("Use the pages on the left:\n- Live Capture\n- PCAP Upload")

# # st.markdown("""
# # - This app demonstrates a streaming anomaly detector with an optional Quantum Kernel SVC trained from your feedback.
# # - A simulated QKD (BB84) key exchange derives a session key used to encrypt sensitive logs/messages via AES-GCM.
# # """)

# # st.info("Go to the left sidebar and open 'Live Capture' to start sniffing, or 'PCAP Upload' to analyze a file.")

# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import streamlit as st

# st.set_page_config(page_title="Quantum Anomaly Detection (QKD-secured)", layout="wide")

# # Style
# try:
#     with open(os.path.join(os.path.dirname(__file__), "assets", "styles.css"), "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# except Exception:
#     pass

# st.markdown("""
# <div class="banner">
#   <h2 style="margin:0;">Quantum Kernel Anomaly Detection</h2>
#   <div class="small">Real-time traffic · Online learning · Quantum vs Classical · QKD-secured pipeline</div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# - Use the pages on the left:
#   - Live Capture
#   - PCAP Upload
#   - Quantum vs Classical
#   - Crypto & QKD
#   - Security Dashboard
# """)

# st.info("Tip: Label samples in Live Capture or PCAP Upload, then open 'Quantum vs Classical' to compare models.")

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from quantum_anomaly.ui import theme

st.set_page_config(page_title="Quantum Anomaly Detection (QKD-secured)", layout="wide", page_icon="🚀")

theme.load_theme_css()
theme.top_navbar(
    team=["Saharsh", "Zubair", "Rakesh", "Arsalan"],
    institute="IIIT Dharwad",
    project="Mini Project"
)
theme.hero(
    title="Real-time Quantum Anomaly Detection",
    subtitle="Online learning • PennyLane Quantum Kernel • Cyber Security Ops • QKD-secured AES-GCM pipeline",
    lottie_url="https://lottie.host/6e02a67a-d0a8-41c8-80da-7df9c934a0b1/7dB9z6v3Nb.json"
)

# CTA cards
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown("""<div class="card"><h3>Live Capture</h3><p class="small">Sniff packets, compute features, stream anomaly scores, and label in real time.</p><a href="/01_Live_Capture" class="cta">Open</a></div>""", unsafe_allow_html=True)
with c2: st.markdown("""<div class="card"><h3>PCAP Upload</h3><p class="small">Analyze PCAPs offline; add labels to train QSVM.</p><a href="/02_PCAP_Upload" class="cta">Open</a></div>""", unsafe_allow_html=True)
with c3: st.markdown("""<div class="card"><h3>Quantum vs Classical</h3><p class="small">Compare QSVM vs RBF-SVM with Accuracy, F1, AUC, ROC & PR curves.</p><a href="/03_Quantum_vs_Classical" class="cta">Compare</a></div>""", unsafe_allow_html=True)
with c4: st.markdown("""<div class="card"><h3>Crypto & QKD</h3><p class="small">Simulate BB84, derive session keys, and encrypt via AES-GCM.</p><a href="/04_Crypto_QKD" class="cta">Explore</a></div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'/>", unsafe_allow_html=True)

# Highlights section
st.subheader("Highlights")
g1, g2, g3 = st.columns(3)
with g1: st.markdown("""<div class="metric"><h3>Quantum Kernel</h3><div class="value">Nyström + PennyLane</div><div class="small">Approximate quantum feature map for online retraining</div></div>""", unsafe_allow_html=True)
with g2: st.markdown("""<div class="metric"><h3>Streaming ML</h3><div class="value">HalfSpaceTrees</div><div class="small">Unsupervised anomaly scoring by River</div></div>""", unsafe_allow_html=True)
with g3: st.markdown("""<div class="metric"><h3>Secured</h3><div class="value">QKD → AES-GCM</div><div class="small">BB84-derived session key, authenticated encryption</div></div>""", unsafe_allow_html=True)

theme.footer()