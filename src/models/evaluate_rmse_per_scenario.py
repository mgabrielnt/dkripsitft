from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

BASELINE_DIR = r"D:/skripsi/tft/models/baseline"
HYBRID_DIR = r"D:/skripsi/tft/models/hybrid"
BASELINE_OUT = r"D:/skripsi/tft/models/baseline/baseline_vallos.csv"
HYBRID_OUT = r"D:/skripsi/tft/models/hybrid/hybrid_vallos.csv"
DEBUG_ONE_ONLY = False


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def list_scenarios(model_dir: str):
    dirs = [d for d in Path(model_dir).iterdir() if d.is_dir() and d.name.upper().startswith("S")]
    return sorted(dirs, key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 10**9)


def find_ckpt(scenario_dir: Path):
    for p in [scenario_dir / "best-checkpoint.ckpt", scenario_dir / "best_model.ckpt"]:
        if p.exists():
            return p
    files = sorted(scenario_dir.glob("*.ckpt"))
    bests = [p for p in files if "best" in p.name.lower()]
    return bests[0] if bests else (files[0] if files else None)


def to_float(x):
    try:
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return np.nan


def first_value(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_ckpt_dict(ckpt_path: Path):
    if ckpt_path is None or not ckpt_path.exists():
        return {}
    try:
        try:
            return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        return {}


def extract_hparams(ckpt_raw: dict):
    hp = ckpt_raw.get("hyper_parameters", {})
    hp2 = ckpt_raw.get("hparams", {})
    out = {}
    if isinstance(hp, dict):
        out.update(hp)
    if isinstance(hp2, dict):
        out.update(hp2)
    return out


def collect_best_scores(obj):
    found = []
    if isinstance(obj, dict):
        if "best_model_score" in obj:
            found.append((str(obj.get("monitor", "")).lower(), to_float(obj.get("best_model_score"))))
        for v in obj.values():
            found.extend(collect_best_scores(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(collect_best_scores(v))
    return found


def read_vallos(scenario_dir: Path, params: dict, ckpt_raw: dict):
    for k in ["best_val_loss", "val_loss", "best_model_score"]:
        v = to_float(params.get(k))
        if np.isfinite(v):
            return float(v)

    for monitor, score in collect_best_scores(ckpt_raw.get("callbacks", {})):
        if monitor == "val_loss" and np.isfinite(score):
            return float(score)
    for _, score in collect_best_scores(ckpt_raw.get("callbacks", {})):
        if np.isfinite(score):
            return float(score)

    for csv_path in sorted(scenario_dir.rglob("metrics.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "val_loss" in df.columns:
            s = pd.to_numeric(df["val_loss"], errors="coerce").dropna()
            if not s.empty:
                return float(s.min())

    return np.nan

def scenario_to_params(scenario_name: str):
    lrs = [0.0002, 0.0005, 0.0007]
    hss = [64, 96, 128]
    bss = [32, 64, 128]

    n = int(str(scenario_name).upper().replace("S", "")) - 1
    lr = lrs[n // 9]
    hs = hss[(n % 9) // 3]
    bs = bss[n % 3]
    return lr, hs, bs

def evaluate_model_dir(model_dir: str, out_csv: str):
    rows = []

    for i, sc_dir in enumerate(list_scenarios(model_dir), start=1):
        if DEBUG_ONE_ONLY and i > 1:
            break

        params = read_json(sc_dir / "params.json")
        ckpt = find_ckpt(sc_dir)
        ckpt_raw = load_ckpt_dict(ckpt)
        hp = extract_hparams(ckpt_raw)

        lr = first_value(params, ["learning_rate", "lr"], None)
        if lr is None:
            lr = first_value(hp, ["learning_rate", "lr"], None)

        hs = first_value(params, ["hidden_size", "hidden_dim", "hidden"], None)
        if hs is None:
            hs = first_value(hp, ["hidden_size", "hidden_continuous_size", "hidden_dim", "hidden"], None)

        bs = first_value(params, ["batch_size", "batch"], None)
        if bs is None:
            bs = first_value(hp, ["batch_size", "batch"], None)

        lr, hs, bs = scenario_to_params(sc_dir.name)

        rows.append({
            "Skenario": sc_dir.name,
            "Learning Rate": lr,
            "Hidden Size": hs,
            "Batch Size": bs,
            "vallos": read_vallos(sc_dir, params, ckpt_raw),
        })

    out = pd.DataFrame(rows).sort_values(["vallos", "Skenario"], na_position="last").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")
    print(out.to_string(index=False))


def main():
    evaluate_model_dir(BASELINE_DIR, BASELINE_OUT)
    evaluate_model_dir(HYBRID_DIR, HYBRID_OUT)


if __name__ == "__main__":
    main()