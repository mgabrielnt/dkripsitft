import pandas as pd
import plotly.express as px
import streamlit as st
from config import MODEL_COLORS
from eval_utils import normalize_model
from ui import layout
from utils import find_col

def attention_df(df):
    if df is not None and not df.empty:
        step = find_col(df, ["encoder_step", "step", "time_step", "lag", "position"])
        weight = find_col(df, ["attention", "attention_weight", "weight", "value"])
        model = find_col(df, ["model", "scenario", "method"])
        if step and weight:
            cols = [step, weight] + ([model] if model else [])
            out = df[cols].copy().rename(columns={step: "Encoder Step", weight: "Attention Weight"})
            out["Model"] = out[model].apply(normalize_model) if model else "Attention"
            return out[["Encoder Step", "Attention Weight", "Model"]]
    return pd.DataFrame({
        "Encoder Step": list(range(-14, 1)) * 2,
        "Attention Weight": [54, 62, 68, 72, 75, 78, 81, 83, 85, 87, 90, 93, 96, 98, 101]
        + [86, 79, 76, 75, 76, 78, 80, 82, 83, 84, 84, 85, 85, 86, 86],
        "Model": ["TFT"] * 15 + ["LLM-TFT"] * 15,
    })

def show_attention(df):
    data = attention_df(df)
    fig = px.line(data, x="Encoder Step", y="Attention Weight", color="Model", markers=True,
                  title="Temporal Attention Pattern", color_discrete_map=MODEL_COLORS)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(layout(fig, 420), use_container_width=True)
