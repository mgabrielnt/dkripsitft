import logging, warnings
import numpy as np, pandas as pd, torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

warnings.filterwarnings("ignore")
for n in ["lightning", "lightning.pytorch", "pytorch_lightning", "pytorch_forecasting"]:
    logging.getLogger(n).setLevel(logging.ERROR)

FEATURE_ALIASES = {
    "vol_20'": "vol_20",
    "sentiiment_delta_1d": "sentiment_delta_1d",
    "log_returrn_2d": "log_return_2d",
    "log_return__2d": "log_return_2d",
}

def load_model(ckpt_path: str):
    try:
        m = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        m = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")
    m.eval(); return m

def get_dataset_parameters(model):
    for obj in [getattr(model, "hparams", None), model]:
        for attr in ["dataset_parameters", "_dataset_parameters"]:
            val = getattr(obj, attr, None) if obj is not None else None
            if isinstance(val, dict) and val: return val
        if isinstance(obj, dict) and isinstance(obj.get("dataset_parameters"), dict):
            return obj["dataset_parameters"]
    raise ValueError("dataset_parameters tidak ditemukan di checkpoint.")

def normalize_df(df: pd.DataFrame):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    for wrong, right in FEATURE_ALIASES.items():
        if right in df.columns and wrong not in df.columns: df[wrong] = df[right]
    if "date" in df.columns: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["ticker", "month", "day_of_week", "is_month_end", "split"]:
        if c in df.columns: df[c] = df[c].astype(str)
    if "time_idx" in df.columns:
        df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
        df = df.dropna(subset=["time_idx"]).copy(); df["time_idx"] = df["time_idx"].astype(int)
    num = df.select_dtypes(include=[np.number]).columns
    if len(num): df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sort_cols = [c for c in ["ticker", "time_idx"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)

def split_df(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    df = normalize_df(df)
    uniq = set(df["split"].astype(str).unique()) if "split" in df.columns else set()
    if {"train", "val", "test"}.issubset(uniq):
        return tuple(df[df["split"] == s].copy() for s in ["train", "val", "test"])
    out = []
    for _, g in df.groupby("ticker", sort=False):
        i1, i2 = int(len(g) * train_ratio), int(len(g) * (train_ratio + val_ratio))
        for s, part in zip(["train", "val", "test"], [g.iloc[:i1], g.iloc[i1:i2], g.iloc[i2:]]):
            part = part.copy(); part["split"] = s; out.append(part)
    out = pd.concat(out, ignore_index=True)
    return tuple(out[out["split"] == s].copy() for s in ["train", "val", "test"])

def _is_internal(c):
    return isinstance(c, str) and (
        c.startswith("__group_id__") or c in {"relative_time_idx", "encoder_length", "decoder_length"}
        or c.endswith("_center") or c.endswith("_scale")
    )

def _raw_cols(dp, keys):
    cols = set()
    for k in keys:
        vals = dp.get(k)
        if isinstance(vals, (list, tuple)): cols |= {c for c in vals if isinstance(c, str) and not _is_internal(c)}
    return sorted(cols)

def _fix_missing(df: pd.DataFrame, missing):
    df = df.copy(); need = {FEATURE_ALIASES.get(c, c) for c in missing}
    if "log_return_2d" in need and "log_return_2d" not in df.columns and "close" in df.columns:
        s = pd.to_numeric(df["close"], errors="coerce")
        df["log_return_2d"] = s.groupby(df["ticker"]).transform(lambda x: np.log(x / x.shift(2))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "sentiment_delta_1d" in need and "sentiment_delta_1d" not in df.columns:
        if {"ticker", "sentiment_final_mean"}.issubset(df.columns):
            base = pd.to_numeric(df["sentiment_final_mean"], errors="coerce")
            df["sentiment_delta_1d"] = base.groupby(df["ticker"]).diff().fillna(0.0)
        else:
            df["sentiment_delta_1d"] = 0.0
    for wrong, right in FEATURE_ALIASES.items():
        if right in df.columns and wrong not in df.columns: df[wrong] = df[right]
    return df

def prepare_eval_df(df: pd.DataFrame, model):
    df, dp = normalize_df(df), get_dataset_parameters(model)
    req = _raw_cols(dp, [
        "static_categoricals", "static_reals", "time_varying_known_categoricals",
        "time_varying_known_reals", "time_varying_unknown_categoricals",
        "time_varying_unknown_reals", "categoricals", "reals", "x_categoricals",
        "x_reals", "group_ids"
    ])
    for k in ["target", "time_idx"]:
        v = dp.get(k)
        if isinstance(v, str) and not _is_internal(v): req.append(v)
    req = sorted(set(req)); miss = [c for c in req if c not in df.columns]
    if miss: df = _fix_missing(df, miss)
    miss = [c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Kolom checkpoint masih hilang: {miss}")
    for c in _raw_cols(dp, ["static_categoricals", "time_varying_known_categoricals", "time_varying_unknown_categoricals", "categoricals", "x_categoricals", "group_ids"]):
        if c in df.columns: df[c] = df[c].astype(str).replace({"nan": "NA", "None": "NA", "<NA>": "NA"}).astype("category")
    df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce").fillna(0).astype(int)
    return df

def predict_bundle(model, df_eval: pd.DataFrame, batch_size=64, mode="prediction"):
    ds = TimeSeriesDataSet.from_parameters(get_dataset_parameters(model), data=prepare_eval_df(df_eval, model), stop_randomization=True, predict=False)
    loader = ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    return model.predict(loader, mode=mode, return_x=True, return_y=True, return_index=True)

def _to_np(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    if isinstance(x, (list, tuple)):
        arrs = [_to_np(v) for v in x]; arrs = [a for a in arrs if a is not None]
        return np.concatenate(arrs, axis=0) if arrs else None
    return None

def _as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 3: x = x[..., x.shape[-1] // 2]
    return x.reshape(-1, 1) if x.ndim == 1 else x

def bundle_arrays(bundle):
    pred = _as_2d(_to_np(getattr(bundle, "prediction", getattr(bundle, "output", bundle))))
    y = getattr(bundle, "y", None); act = _as_2d(_to_np(y[0] if isinstance(y, (tuple, list)) else y))
    enc = _to_np(getattr(bundle, "x", {}).get("encoder_target"))
    anchor = _as_2d(enc)[..., -1].reshape(-1, 1) if enc is not None else np.full((len(pred), 1), np.nan)
    n, h = min(len(pred), len(act), len(anchor)), min(pred.shape[1], act.shape[1])
    return pred[:n, :h], act[:n, :h], anchor[:n]

def _metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0: return dict(rmse=np.nan, mae=np.nan, mape=np.nan, r2=np.nan, n=0)
    out = {"rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))), "mae": float(np.mean(np.abs(y_true - y_pred))), "n": int(len(y_true))}
    nz = np.abs(y_true) > 1e-8
    out["mape"] = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100.0) if nz.any() else np.nan
    ss = np.sum((y_true - np.mean(y_true)) ** 2)
    out["r2"] = float(1 - np.sum((y_true - y_pred) ** 2) / ss) if ss != 0 else np.nan
    return out

def _diracc(pred, act, anchor, eps=0.0):
    pred, act, anchor = map(lambda z: np.asarray(z, dtype=float), [pred, act, anchor])
    valid = np.isfinite(pred) & np.isfinite(act) & np.isfinite(anchor)
    act_move, pred_move = act - anchor, pred - anchor
    if eps > 0: valid &= np.abs(act_move) > eps
    if valid.sum() == 0: return np.nan, 0
    return float((np.sign(pred_move[valid]) == np.sign(act_move[valid])).mean() * 100.0), int(valid.sum())

def evaluate_bundle(bundle, diracc_eps=0.0):
    pred2d, act2d, anchor = bundle_arrays(bundle)
    g = _metrics(act2d.reshape(-1), pred2d.reshape(-1))
    g["diracc"], g["n_diracc"] = _diracc(pred2d.reshape(-1), act2d.reshape(-1), np.repeat(anchor.reshape(-1), pred2d.shape[1]), diracc_eps)
    rows = []
    for i in range(pred2d.shape[1]):
        r = _metrics(act2d[:, i], pred2d[:, i]); r["horizon"] = f"H{i+1}"
        r["diracc"], r["n_diracc"] = _diracc(pred2d[:, i], act2d[:, i], anchor[:, 0], diracc_eps); rows.append(r)
    return g, pd.DataFrame(rows)[["horizon", "rmse", "mae", "mape", "r2", "diracc", "n", "n_diracc"]]