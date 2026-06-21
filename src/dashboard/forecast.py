import pandas as pd
from config import CHECKPOINTS
from utils import date_col, find_col, find_contains_col, norm
from forecast_model import load_tft, make_dataset

MODEL_COLS = ["model", "model_name", "scenario", "skenario", "method", "metode", "variant", "architecture"]
HORIZON_COLS = ["horizon", "h", "step", "forecast_horizon", "prediction_horizon"]
PRED_COLS = ["prediction", "predicted", "y_pred", "pred", "forecast", "pred_close"]
LLM_KEYS = ["llmtft", "llm", "hybrid", "sent", "s1"]
TFT_KEYS = ["tft", "baseline", "base", "s5"]


def normalize_model(value):
    low = str(value).lower()
    if any(key in low for key in ["llm", "hybrid", "sent", "s1"]):
        return "LLM-TFT"
    if any(key in low for key in ["tft", "base", "s5"]):
        return "TFT"
    return str(value)


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


def extract_predictions(df, preferred_model="LLM-TFT"):
    result = {"H+1": None, "H+2": None, "H+3": None}
    if df is None or df.empty:
        return result
    work = prepare_prediction_frame(df, preferred_model)
    result = extract_model_wide_predictions(work, result)
    if any(v is not None for v in result.values()):
        return result
    result = extract_wide_predictions(work, result)
    return result if any(v is not None for v in result.values()) else extract_long_predictions(work, result)


def prepare_prediction_frame(df, preferred_model):
    work = df.sort_values(date_col(df)) if date_col(df) else df.copy()
    model_col = find_col(work, MODEL_COLS)
    if model_col:
        work = work.assign(_series=work[model_col].apply(normalize_model))
        chosen = work[work["_series"].eq(preferred_model)]
        if not chosen.empty:
            return chosen
    return work


def extract_model_wide_predictions(work, result):
    for step in [1, 2, 3]:
        col = find_model_prediction_col(work, step, LLM_KEYS)
        if col and pd.api.types.is_numeric_dtype(work[col]):
            val = work[col].dropna().tail(1)
            result[f"H+{step}"] = None if val.empty else float(val.iloc[-1])
    return result


def find_model_prediction_col(work, step, model_keys):
    horizon_keys = [f"h{step}", f"horizon{step}", f"pred{step}", f"prediction{step}", f"{step}"]
    pred_keys = ["pred", "prediction", "forecast", "yhat", "ypred"]
    for col in work.columns:
        low = norm(col)
        has_model = any(key in low for key in model_keys)
        has_horizon = any(key in low for key in horizon_keys)
        has_pred = any(key in low for key in pred_keys)
        if has_model and has_horizon and has_pred:
            return col
    return None


def extract_wide_predictions(work, result):
    wide = {
        "H+1": ["llm_tft_pred_h1", "llm_tft_h1", "hybrid_pred_h1", "llm_h1", "pred_h1", "prediction_h1", "h1_pred", "H+1"],
        "H+2": ["llm_tft_pred_h2", "llm_tft_h2", "hybrid_pred_h2", "llm_h2", "pred_h2", "prediction_h2", "h2_pred", "H+2"],
        "H+3": ["llm_tft_pred_h3", "llm_tft_h3", "hybrid_pred_h3", "llm_h3", "pred_h3", "prediction_h3", "h3_pred", "H+3"],
    }
    for horizon, cols in wide.items():
        col = find_col(work, cols)
        if col and pd.api.types.is_numeric_dtype(work[col]):
            val = work[col].dropna().tail(1)
            result[horizon] = None if val.empty else float(val.iloc[-1])
    return result


def extract_long_predictions(work, result):
    hcol = find_col(work, HORIZON_COLS)
    pcol = find_col(work, PRED_COLS) or find_contains_col(work, ["pred"], ["error"])
    if not (hcol and pcol):
        return result
    tmp = work[[hcol, pcol]].dropna().copy()
    tmp[hcol] = pd.to_numeric(tmp[hcol].astype(str).str.replace("H+", "", regex=False), errors="coerce")
    for i in [1, 2, 3]:
        val = tmp[tmp[hcol].eq(i)][pcol].dropna().tail(1)
        result[f"H+{i}"] = None if val.empty else float(val.iloc[-1])
    return result


def prediction_rows(checkpoint_preds, fallback):
    rows = checkpoint_rows(checkpoint_preds)
    has_llm = any(row["Series"] == "LLM-TFT" for row in rows)
    fallback_rows = [{"Series": "LLM-TFT", "Step": int(h[-1]), "Harga": v} for h, v in fallback.items() if v is not None]
    if fallback_rows and not has_llm:
        rows.extend(fallback_rows)
    return pd.DataFrame(rows)


def checkpoint_rows(checkpoint_preds):
    rows = []
    for model, values in checkpoint_preds.items():
        series = model.replace(" S5", "").replace(" S1", "")
        for h, val in zip(["H+1", "H+2", "H+3"], values):
            if val is not None:
                rows.append({"Series": series, "Step": int(h[-1]), "Harga": val})
    return rows
