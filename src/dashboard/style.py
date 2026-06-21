import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
    radial-gradient(circle at top left, rgba(56,189,248,0.17), transparent 34%),
    radial-gradient(circle at top right, rgba(249,115,22,0.13), transparent 30%),
    linear-gradient(135deg, #050B16 0%, #0F172A 48%, #111827 100%);
    color: #e5e7eb;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050B16 0%, #0B1220 100%);
    border-right: 1px solid rgba(148,163,184,0.18);
}
.main-title {
    padding: 24px 28px; border-radius: 26px;
    background:
    radial-gradient(circle at 10% 0%, rgba(56,189,248,0.34), transparent 34%),
    radial-gradient(circle at 95% 20%, rgba(249,115,22,0.22), transparent 30%),
    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.72));
    border: 1px solid rgba(148,163,184,0.25);
    box-shadow: 0 22px 60px rgba(0,0,0,0.30);
    margin-bottom: 22px;
}
.main-title h1 { margin: 0; font-size: 2.05rem; letter-spacing: -0.04em; color: #f8fafc; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(15,23,42,0.88), rgba(30,41,59,0.70));
    border: 1px solid rgba(148,163,184,0.18);
    padding: 15px 16px; border-radius: 18px; box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
</style>
"""

def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)

def header(title):
    st.markdown(f"<div class='main-title'><h1>{title}</h1></div>", unsafe_allow_html=True)
