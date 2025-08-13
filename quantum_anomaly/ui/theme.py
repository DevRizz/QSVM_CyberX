import os
import streamlit as st

def load_theme_css():
    css_path = os.path.join(os.path.dirname(__file__), "..", "..", "streamlit_app", "assets", "styles.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def top_navbar(team: list[str] | None = None, institute: str = "IIIT Dharwad", project: str = "Mini Project"):
    team = team or []
    members = " • ".join(team) if team else "Add your team in theme.top_navbar()"
    html = f"""
    <div class="navbar">
      <span class="badge">{institute}</span>
      <div class="brand">
        <svg width="26" height="26" viewBox="0 0 100 100" fill="none">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#7928ca"/><stop offset="100%" stop-color="#ff0080"/>
            </linearGradient>
          </defs>
          <circle cx="50" cy="50" r="46" stroke="url(#g)" stroke-width="6" fill="rgba(255,255,255,0.04)"/>
          <path d="M30 62 L50 22 L70 62 Z" fill="url(#g)"/>
        </svg>
        <h2>Quantum Anomaly Detection</h2>
      </div>
      <span class="badge" style="margin-left:auto;">{project}</span>
    </div>
    <div class="small" style="margin:8px 4px 18px 4px;">Team: {members}</div>
    """
    st.markdown(html, unsafe_allow_html=True)

def hero(title: str, subtitle: str, lottie_url: str | None = None):
    st.markdown(f"""
    <div class="hero">
      <h1>{title}</h1>
      <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)
    if lottie_url:
        st.markdown("""
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="lottie" style="margin-top:10px;">
          <lottie-player src="{lottie_url}" background="transparent" speed="1" style="width:100%;height:220px;" loop autoplay></lottie-player>
        </div>
        """, unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class="footer">
      <span>© 2025 IIIT Dharwad — Quantum Kernel Online Anomaly Detection with QKD-secured pipeline</span>
    </div>
    """, unsafe_allow_html=True)