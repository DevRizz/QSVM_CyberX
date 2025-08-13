# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from quantum_anomaly.orchestrator import Orchestrator
# from quantum_anomaly.models.classical_svm import ClassicalSVM
# from quantum_anomaly.models.eval import train_test_metrics

# st.set_page_config(page_title="Quantum vs Classical", layout="wide")
# st.title("Quantum vs Classical: Model Comparison")

# # Style
# try:
#     with open(os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css"), "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# except Exception:
#     pass

# if "orchestrator" not in st.session_state:
#     st.session_state["orchestrator"] = Orchestrator()
# orch: Orchestrator = st.session_state["orchestrator"]

# st.markdown('<div class="banner">Compare Quantum Kernel SVC (Nyström) against Classical SVM (RBF) on your labeled buffer.</div>', unsafe_allow_html=True)
# st.caption("Tip: Provide labels via Live Capture or PCAP Upload pages. Once you have enough labeled samples, evaluate here.")

# # Collect labeled data from buffer
# X_lab, y_lab = orch.buffer.labeled()
# if X_lab.ndim == 1 or len(X_lab) < 20 or len(set(y_lab.tolist())) < 2:
#     st.warning("Need at least 20 labeled samples with both classes present to evaluate. Label more data first.")
#     st.stop()

# qsvc = orch.qsvc.__class__(n_landmarks=int(os.getenv("QK_LANDMARKS","64")))  # fresh instance for fair split
# csvm = ClassicalSVM()

# res = train_test_metrics(X_lab, y_lab, qsvc, csvm, test_size=0.3, seed=42)

# # Metric cards
# cols = st.columns(6)
# cols[0].metric("QSVM Acc", f"{res['quantum']['accuracy']:.3f}")
# cols[1].metric("QSVM F1", f"{res['quantum']['f1']:.3f}")
# cols[2].metric("QSVM AUC", f"{res['quantum']['auc']:.3f}")
# cols[3].metric("Classical Acc", f"{res['classical']['accuracy']:.3f}")
# cols[4].metric("Classical F1", f"{res['classical']['f1']:.3f}")
# cols[5].metric("Classical AUC", f"{res['classical']['auc']:.3f}")

# st.markdown(f'<div class="small">Train size: {res["n_train"]} • Test size: {res["quantum"]["n_test"]}</div>', unsafe_allow_html=True)

# c1, c2 = st.columns(2)

# with c1:
#     st.subheader("ROC Curve")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=res["quantum"]["roc_curve"]["fpr"], y=res["quantum"]["roc_curve"]["tpr"], mode="lines", name="Quantum"))
#     fig.add_trace(go.Scatter(x=res["classical"]["roc_curve"]["fpr"], y=res["classical"]["roc_curve"]["tpr"], mode="lines", name="Classical"))
#     fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash", color="#888")))
#     fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", template="plotly_dark")
#     st.plotly_chart(fig, use_container_width=True)

# with c2:
#     st.subheader("Precision-Recall Curve")
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=res["quantum"]["pr_curve"]["recall"], y=res["quantum"]["pr_curve"]["precision"], mode="lines", name="Quantum"))
#     fig2.add_trace(go.Scatter(x=res["classical"]["pr_curve"]["recall"], y=res["classical"]["pr_curve"]["precision"], mode="lines", name="Classical"))
#     fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
#     st.plotly_chart(fig2, use_container_width=True)

# c3, c4 = st.columns(2)
# with c3:
#     st.subheader("Confusion Matrix (Quantum)")
#     st.write(res["quantum"]["confusion"])
# with c4:
#     st.subheader("Confusion Matrix (Classical)")
#     st.write(res["classical"]["confusion"])

# st.divider()
# st.markdown("""
# - QSVM uses a quantum kernel approximation (Nyström with PennyLane) to map features into a quantum Hilbert space.
# - Classical SVM uses an RBF kernel on the original scaled features.
# - Depending on the traffic pattern, QSVM may capture non-classical similarities and improve separability.
# """)

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from quantum_anomaly.ui import theme
from quantum_anomaly.orchestrator import Orchestrator
from quantum_anomaly.models.classical_svm import ClassicalSVM
from quantum_anomaly.models.eval import train_test_metrics

st.set_page_config(page_title="Quantum vs Classical", layout="wide", page_icon="⚖️")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Quantum vs Classical",
    "Compare QSVM (Nyström quantum kernel) and classical RBF-SVM with robust metrics.",
    lottie_url="https://lottie.host/f7bdb1e5-9d8d-4dc0-9c3d-9a1f47cda5e7/2y6MBJ3m2w.json"
)

if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()
orch: Orchestrator = st.session_state["orchestrator"]

X_lab, y_lab = orch.buffer.labeled()
if X_lab.ndim == 1 or len(X_lab) < 20 or len(set(y_lab.tolist())) < 2:
    st.warning("Need at least 20 labeled samples with both classes present to evaluate. Label more data first.")
    st.stop()

qsvc = orch.qsvc.__class__(n_landmarks=int(os.getenv("QK_LANDMARKS","64")))
csvm = ClassicalSVM()

res = train_test_metrics(X_lab, y_lab, qsvc, csvm, test_size=0.3, seed=42)

cols = st.columns(6)
cols[0].markdown(f'<div class="metric"><h3>QSVM Accuracy</h3><div class="value">{res["quantum"]["accuracy"]:.3f}</div></div>', unsafe_allow_html=True)
cols[1].markdown(f'<div class="metric"><h3>QSVM F1</h3><div class="value">{res["quantum"]["f1"]:.3f}</div></div>', unsafe_allow_html=True)
cols[2].markdown(f'<div class="metric"><h3>QSVM AUC</h3><div class="value">{res["quantum"]["auc"]:.3f}</div></div>', unsafe_allow_html=True)
cols[3].markdown(f'<div class="metric"><h3>RBF Accuracy</h3><div class="value">{res["classical"]["accuracy"]:.3f}</div></div>', unsafe_allow_html=True)
cols[4].markdown(f'<div class="metric"><h3>RBF F1</h3><div class="value">{res["classical"]["f1"]:.3f}</div></div>', unsafe_allow_html=True)
cols[5].markdown(f'<div class="metric"><h3>RBF AUC</h3><div class="value">{res["classical"]["auc"]:.3f}</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="small">Train size: {res["n_train"]} • Test size: {res["quantum"]["n_test"]}</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("ROC Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["quantum"]["roc_curve"]["fpr"], y=res["quantum"]["roc_curve"]["tpr"], mode="lines", name="QSVM"))
    fig.add_trace(go.Scatter(x=res["classical"]["roc_curve"]["fpr"], y=res["classical"]["roc_curve"]["tpr"], mode="lines", name="RBF"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash", color="#666")))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", template="plotly_dark", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.subheader("Precision-Recall Curve")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=res["quantum"]["pr_curve"]["recall"], y=res["quantum"]["pr_curve"]["precision"], mode="lines", name="QSVM"))
    fig2.add_trace(go.Scatter(x=res["classical"]["pr_curve"]["recall"], y=res["classical"]["pr_curve"]["precision"], mode="lines", name="RBF"))
    fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark", legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr class='divider'/>", unsafe_allow_html=True)
st.markdown("""
- QSVM leverages a quantum-inspired kernel map; RBF-SVM works in classical feature space.
- Differences in ROC/PR indicate which model better separates classes under your traffic pattern.
""")