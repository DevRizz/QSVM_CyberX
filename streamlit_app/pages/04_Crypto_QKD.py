# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import json
# import streamlit as st
# from quantum_anomaly.security.qkd import bb84_simulate_key, derive_session_key
# from quantum_anomaly.security.secure_channel import SecureChannel

# st.set_page_config(page_title="Crypto & QKD", layout="wide")
# st.title("Crypto & QKD")

# # Style
# try:
#     with open(os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css"), "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# except Exception:
#     pass

# st.markdown('<div class="banner">Simulate BB84, observe QBER, derive a session key, and try AES-GCM encryption.</div>', unsafe_allow_html=True)

# colA, colB = st.columns([1,1])

# with colA:
#     st.subheader("BB84 Simulation")
#     length = st.slider("Key length (qubits)", min_value=256, max_value=4096, value=2048, step=256)
#     err = st.slider("Eavesdropping / channel error rate", min_value=0.0, max_value=0.2, value=0.01, step=0.005)
#     seed = st.number_input("Seed (optional)", min_value=0, value=0, step=1)
#     use_seed = st.checkbox("Use seed", value=False)
#     if st.button("Run BB84"):
#         raw, qber, sifted_len = bb84_simulate_key(length=length, seed=(seed if use_seed else None), error_rate=err)
#         st.session_state["bb84"] = {"raw": raw, "qber": qber, "sifted": sifted_len}

#     if "bb84" in st.session_state:
#         s = st.session_state["bb84"]
#         st.metric("Sifted bits", s["sifted"])
#         st.metric("Observed QBER", f"{s['qber']:.3f}")
#         if s["qber"] < 0.11:
#             st.success("QBER below threshold. Proceed to key derivation.")
#         else:
#             st.error("QBER too high — key discarded (possible MITM or noisy channel).")

# with colB:
#     st.subheader("Key Derivation + AES-GCM")
#     if "bb84" in st.session_state and st.session_state["bb84"]["qber"] < 0.11:
#         raw = st.session_state["bb84"]["raw"]
#         key = derive_session_key(raw, out_len=32)
#         st.write("Session key (hex, first 32 chars):", key.hex()[:32] + "…")
#         chan = SecureChannel(key)
#         msg = st.text_area("Message to encrypt", value="Hello from QKD-secured channel!")
#         aad = st.text_input("Associated data (AAD)", value="meta")
#         if st.button("Encrypt & Decrypt"):
#             nonce, ct = chan.encrypt(msg.encode("utf-8"), aad=aad.encode("utf-8"))
#             st.write("Nonce (hex):", nonce.hex())
#             st.write("Ciphertext (hex):", ct.hex())
#             pt = chan.decrypt(nonce, ct, aad=aad.encode("utf-8")).decode("utf-8")
#             st.success(f"Decrypted: {pt}")
#     else:
#         st.info("Run BB84 and ensure QBER < 0.11 to derive a key.")
#         st.warning("QBER too high or no BB84 run yet. Cannot derive key.")
#         st.stop()

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from quantum_anomaly.ui import theme
from quantum_anomaly.security.qkd import bb84_simulate_key, derive_session_key
from quantum_anomaly.security.secure_channel import SecureChannel

st.set_page_config(page_title="Crypto & QKD", layout="wide", page_icon="🔐")
theme.load_theme_css()
theme.top_navbar(team=["Saharsh", "Zubair", "Rakesh", "Arsalan"])
theme.hero(
    "Quantum Key Distribution (BB84) + AES-GCM",
    "Simulate BB84, measure QBER, derive a session key, and encrypt messages with AEAD.",
    lottie_url="https://lottie.host/9dc5f160-82f7-4dbe-8d81-0a7b4a3f9928/0f93fISwby.json"
)

colA, colB = st.columns([1,1])
with colA:
    st.subheader("BB84 Simulation")
    length = st.slider("Key length (qubits)", 256, 4096, 2048, 256)
    err = st.slider("Channel error / eavesdropping rate", 0.0, 0.2, 0.01, 0.005)
    use_seed = st.checkbox("Use seed", value=False)
    seed = st.number_input("Seed", min_value=0, value=0, step=1, disabled=not use_seed)
    if st.button("Run BB84", type="primary"):
        raw, qber, sifted = bb84_simulate_key(length=length, seed=(seed if use_seed else None), error_rate=err)
        st.session_state["bb84"] = {"raw": raw, "qber": qber, "sifted": sifted}
    if "bb84" in st.session_state:
        s = st.session_state["bb84"]
        a, b, c = st.columns(3)
        a.metric("Sifted bits", s["sifted"])
        b.metric("Observed QBER", f"{s['qber']:.3f}")
        ok = s["qber"] < 0.11
        c.metric("Key usable", "Yes" if ok else "No")
        if ok: st.success("QBER < 11% — proceed to key derivation.")
        else: st.error("QBER too high — discard key (possible MITM/noisy channel).")

with colB:
    st.subheader("Derive Session Key + AEAD")
    if "bb84" in st.session_state and st.session_state["bb84"]["qber"] < 0.11:
        raw = st.session_state["bb84"]["raw"]
        key = derive_session_key(raw, out_len=32)
        st.write("Session key (hex, first 32 chars): ", key.hex()[:32] + "…")
        chan = SecureChannel(key)
        msg = st.text_area("Message", "Hello from QKD-secured channel!")
        aad = st.text_input("Associated data (AAD)", "meta")
        if st.button("Encrypt & Decrypt", type="secondary"):
            nonce, ct = chan.encrypt(msg.encode(), aad=aad.encode())
            st.code(f"nonce={nonce.hex()}\nciphertext={ct.hex()}", language="text")
            pt = chan.decrypt(nonce, ct, aad=aad.encode()).decode()
            st.success("Decrypted: " + pt)
    else:
        st.info("Run BB84 and ensure QBER < 0.11 to derive a key.")