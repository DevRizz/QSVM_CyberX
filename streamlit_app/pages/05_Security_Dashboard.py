# import os, sys, json, sqlite3
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from quantum_anomaly.orchestrator import Orchestrator

# st.set_page_config(page_title="Security Dashboard", layout="wide")
# st.title("Security Dashboard")

# # Style
# try:
#     with open(os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css"), "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# except Exception:
#     pass

# if "orchestrator" not in st.session_state:
#     st.session_state["orchestrator"] = Orchestrator()
# orch: Orchestrator = st.session_state["orchestrator"]

# db_path = os.getenv("DB_PATH", "./anomaly_events.db")
# st.caption(f"Database: {db_path}")

# limit = st.slider("Load last N events", min_value=50, max_value=5000, value=500, step=50)

# @st.cache_data(ttl=5.0)
# def load_events(dbp: str, lim: int) -> pd.DataFrame:
#     con = sqlite3.connect(dbp)
#     df = pd.read_sql_query(f"SELECT * FROM events ORDER BY id DESC LIMIT {lim}", con)
#     con.close()
#     return df

# df = load_events(db_path, limit)
# if df.empty:
#     st.info("No events yet. Start Live Capture or upload a PCAP.")
#     st.stop()

# # Attempt decrypt features_json if it's encrypted and we have the key
# def parse_features(cell: str):
#     try:
#         d = json.loads(cell)
#         if isinstance(d, dict) and "nonce" in d and "ct" in d and orch.secure is not None:
#             nonce = bytes.fromhex(d["nonce"]); ct = bytes.fromhex(d["ct"])
#             pt = orch.secure.decrypt(nonce, ct)
#             return json.loads(pt.decode("utf-8"))
#         return d if isinstance(d, dict) else {}
#     except Exception:
#         return {}

# parsed = df["features_json"].apply(parse_features)
# feat_df = pd.json_normalize(parsed)

# view = pd.concat([df[["ts","src_ip","dst_ip","protocol","anomaly_score","label"]], feat_df], axis=1)
# st.dataframe(view.head(100), use_container_width=True)

# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("Top Source IPs")
#     top = view["src_ip"].value_counts().head(10).reset_index()
#     top.columns = ["src_ip", "events"]
#     st.plotly_chart(px.bar(top, x="src_ip", y="events", template="plotly_dark"), use_container_width=True)

# with col2:
#     st.subheader("Anomaly Score Distribution")
#     st.plotly_chart(px.histogram(view, x="anomaly_score", nbins=40, template="plotly_dark"), use_container_width=True)

# st.subheader("Simple Port Scan Heuristic")
# if "dst_port" in view.columns:
#     thresh = st.slider("Unique ports threshold (per src in last window)", 5, 200, 50, 5)
#     window = min(1000, len(view))
#     recent = view.head(window)
#     scans = (recent.groupby("src_ip")["dst_port"].nunique().sort_values(ascending=False))
#     susp = scans[scans >= thresh]
#     if len(susp):
#         st.warning("Potential port scans detected:")
#         st.dataframe(susp.reset_index().rename(columns={"dst_port":"unique_ports"}))
#     else:
#         st.success("No obvious port scans in the recent window.")
# else:
#     st.info("No dst_port field detected in parsed features.")

import os, sys, json, sqlite3
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import plotly.express as px
from quantum_anomaly.ui import theme
from quantum_anomaly.orchestrator import Orchestrator

st.set_page_config(page_title="Security Dashboard", layout="wide", page_icon="🛡️")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Cyber Security Dashboard",
    "Operational analytics: events, anomalies, top talkers, and port-scan heuristic.",
    lottie_url="https://lottie.host/f7194a01-0a67-4d69-8d82-6a0105f06a7d/BsC2GPRnCr.json"
)

if "orchestrator" not in st.session_state:
    st.session_state["orchestrator"] = Orchestrator()
orch: Orchestrator = st.session_state["orchestrator"]

db_path = os.getenv("DB_PATH", "./anomaly_events.db")
st.caption(f"Database: {db_path}")

limit = st.slider("Load last N events", 50, 5000, 500, 50)

@st.cache_data(ttl=5.0)
def load_events(dbp: str, lim: int) -> pd.DataFrame:
    con = sqlite3.connect(dbp)
    df = pd.read_sql_query(f"SELECT * FROM events ORDER BY id DESC LIMIT {lim}", con)
    con.close()
    return df

df = load_events(db_path, limit)
if df.empty:
    st.info("No events yet. Start Live Capture or upload a PCAP.")
    st.stop()

def parse_features(cell: str):
    try:
        d = json.loads(cell)
        if isinstance(d, dict) and "nonce" in d and "ct" in d and orch.secure is not None:
            nonce = bytes.fromhex(d["nonce"]); ct = bytes.fromhex(d["ct"])
            pt = orch.secure.decrypt(nonce, ct)
            return json.loads(pt.decode("utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

parsed = df["features_json"].apply(parse_features)
feat_df = pd.json_normalize(parsed)

view = pd.concat([df[["ts","src_ip","dst_ip","protocol","anomaly_score","label"]], feat_df], axis=1)
st.dataframe(view.head(100), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Source IPs")
    top = view["src_ip"].value_counts().head(10).reset_index()
    top.columns = ["src_ip", "events"]
    st.plotly_chart(px.bar(top, x="src_ip", y="events", template="plotly_dark"), use_container_width=True)

with col2:
    st.subheader("Anomaly Score Distribution")
    st.plotly_chart(px.histogram(view, x="anomaly_score", nbins=40, template="plotly_dark"), use_container_width=True)

st.subheader("Simple Port Scan Heuristic")
if "dst_port" in view.columns:
    thresh = st.slider("Unique ports threshold (per src in last window)", 5, 200, 50, 5)
    window = min(1000, len(view))
    recent = view.head(window)
    scans = (recent.groupby("src_ip")["dst_port"].nunique().sort_values(ascending=False))
    susp = scans[scans >= thresh]
    if len(susp):
        st.warning("Potential port scans detected:")
        st.dataframe(susp.reset_index().rename(columns={"dst_port":"unique_ports"}))
    else:
        st.success("No obvious port scans in the recent window.")
else:
    st.info("No dst_port field detected in parsed features.")

theme.footer()
