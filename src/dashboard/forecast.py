import pandas as pd
from config import CHECKPOINTS
from utils import date_col, find_col, find_contains_col
from forecast_model import load_tft, make_dataset

def tensor_to_list(output):
    if hasattr(output, "output"):
        output = output.output
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    try:
        return pd.Series(output.reshape(-1)).dropna().astype(float).tolist()[:3]
    except Exception:
        return []

def predict_checkpoints(master, ticker, cutoff):
    out = {}
    if master is None or master.empty:
        return out
    for model, scenario, ckpt in CHECKPOINTS:
        key = f"{model} {scenario}"
        try:
            loaded = load_tft(ckpt)
            dataset = make_dataset(master, ticker, cutoff, model)
            if loaded is None or dataset is None:
                out[key] = [None, None, None]
                continue
            loader = dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            values = tensor_to_list(loaded.predict(loader, mode="prediction", return_x=False))
            out[key] = values + [None] * (3 - len(values))
        except Exception:
            out[key] = [None, None, None]
    return out

def extract_predictions(df):
    result = {"H+1": None, "H+2": None, "H+3": None}
    if df is None or df.empty:
        return result
    work = df.sort_values(date_col(df)) if date_col(df) else df.copy()
    wide = {"H+1": ["pred_h1", "prediction_h1", "h1_pred", "H+1"],
            "H+2": ["pred_h2", "prediction_h2", "h2_pred", "H+2"],
            "H+3": ["pred_h3", "prediction_h3", "h3_pred", "H+3"]}
    for horizon, cols in wide.items():
        col = find_col(work, cols)
        if col and pd.api.types.is_numeric_dtype(work[col]):
            val = work[col].dropna().tail(1)
            result[horizon] = None if val.empty else float(val.iloc[-1])
    if any(v is not None for v in result.values()):
        return result
    hcol = find_col(work, ["horizon", "h", "step", "forecast_horizon"])
    pcol = find_col(work, ["prediction", "predicted", "y_pred", "pred", "forecast"])
    pcol = pcol or find_contains_col(work, ["pred"], ["error"])
    if hcol and pcol:
        tmp = work[[hcol, pcol]].dropna().copy()
        tmp[hcol] = pd.to_numeric(tmp[hcol].astype(str).str.replace("H+", "", regex=False), errors="coerce")
        for i in [1, 2, 3]:
            val = tmp[tmp[hcol].eq(i)][pcol].dropna().tail(1)
            result[f"H+{i}"] = None if val.empty else float(val.iloc[-1])
    return result

def prediction_rows(checkpoint_preds, fallback):
    rows = []
    for model, values in checkpoint_preds.items():
        for h, val in zip(["H+1", "H+2", "H+3"], values):
            if val is not None:
                rows.append({"Series": model.replace(" S5", "").replace(" S1", ""), "Step": int(h[-1]), "Harga": val})
    if not rows:
        rows = [{"Series": "LLM-TFT", "Step": int(h[-1]), "Harga": v} for h, v in fallback.items() if v is not None]
    return pd.DataFrame(rows)
