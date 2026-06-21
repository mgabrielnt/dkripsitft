import subprocess
import streamlit as st
from config import COLORS, ROOT
from commands import PIPELINE

def layout(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.38)",
        margin=dict(l=18, r=18, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb"),
        colorway=COLORS,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)")
    return fig

def run_command(command):
    with st.spinner(f"Menjalankan: {command}"):
        try:
            result = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True, text=True)
        except Exception as exc:
            st.error(f"Gagal menjalankan perintah: {exc}")
            return
    st.success("Proses selesai.") if result.returncode == 0 else st.error("Proses gagal.")
    with st.expander("Log terminal"):
        if result.stdout:
            st.code(result.stdout[-8000:])
        if result.stderr:
            st.code(result.stderr[-8000:])

def action_button(label, key):
    if st.button(label, use_container_width=True):
        run_command(PIPELINE[key])
