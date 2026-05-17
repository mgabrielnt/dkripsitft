from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np, pandas as pd
try:
    from .tft_eval_utils import split_df, load_model, predict_bundle, evaluate_bundle
except ImportError:
    from tft_eval_utils import split_df, load_model, predict_bundle, evaluate_bundle

DATA_PATH=r"D:/skripsi/tft/data/processed/tft_master.csv"; REPORT_DIR=Path(r"D:/skripsi/tft/reportss"); FIG_DIR=REPORT_DIR/"figures"; DIRACC_EPS=15.0
BASELINE_SCENARIO,HYBRID_SCENARIO="S5","S1"; BASELINE_CKPT=rf"D:/skripsi/tft/modelssss/baseline/{BASELINE_SCENARIO}/best-checkpoint.ckpt"; HYBRID_CKPT=rf"D:/skripsi/tft/modelssss/hybrid/{HYBRID_SCENARIO}/best-checkpoint.ckpt"
EVAL_SPLIT,BASE_BATCH_SIZE,HYBRID_BATCH_SIZE="TESTING",128,128

it=lambda s:r'$\it{'+str(s).replace(' ',r'\ ')+r'}$'

def _title_split(): return "VALIDASI" if EVAL_SPLIT=="val" else "TESTING"
def _to_np(x):
    if hasattr(x,"detach"): x=x.detach().cpu().numpy()
    a=np.asarray(x,dtype=float)
    return a[...,0] if a.ndim==3 and a.shape[-1]==1 else a

def eval_per_ticker(model,label,scenario,df_eval,batch_size,eps):
    grows,hdfs=[],[]
    for t in sorted(df_eval["ticker"].astype(str).unique()):
        b=predict_bundle(model,df_eval[df_eval["ticker"].astype(str)==t],batch_size)
        g,h=evaluate_bundle(b,diracc_eps=eps); grows.append({"model":label,"scenario":scenario,"split":EVAL_SPLIT,"ticker":t,**g})
        h.insert(0,"ticker",t); h.insert(0,"split",EVAL_SPLIT); h.insert(0,"scenario",scenario); h.insert(0,"model",label); hdfs.append(h)
    cols=["model","scenario","split","ticker","rmse","mae","mape","r2","diracc","n","n_diracc"]
    return pd.DataFrame(grows)[cols],pd.concat(hdfs,ignore_index=True)

def save_forecast_plot(label, raw_bundle, path, history_window=15, ticker_target="TLKM"):
    x = getattr(raw_bundle, "x", {})
    idx_df = getattr(raw_bundle, "index", None)

    # cari index TLKM
    idx = 0
    if idx_df is not None and "ticker" in idx_df.columns:
        match = idx_df[idx_df["ticker"].astype(str) == ticker_target]
        if len(match) > 0:
            idx = match.index[0]

    out = getattr(raw_bundle, "output", getattr(raw_bundle, "prediction", raw_bundle))
    pred = _to_np(getattr(out, "prediction", out))
    y = getattr(raw_bundle, "y", None)
    act = _to_np(y[0] if isinstance(y, (tuple, list)) else y)

    if pred.ndim == 3:
        pred = pred[..., pred.shape[-1] // 2]

    enc_y_all = _to_np(x["encoder_target"][idx]).reshape(-1)
    act_y = act[idx].reshape(-1)
    pred_y = pred[idx].reshape(-1)

    keep = min(history_window, len(enc_y_all))
    enc_y = enc_y_all[-keep:]
    enc_x = np.arange(-keep + 1, 1)
    dec_x = np.arange(1, len(pred_y) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(enc_x, enc_y, marker="o", linewidth=2.2, label=r"$\it{Actual\ (History)}$")
    ax.plot(dec_x, act_y, marker="o", linewidth=2.2, label=r"$\it{Actual\ (Future)}$")
    ax.plot(dec_x, pred_y, marker="s", linestyle="--", linewidth=2.2, label=r"$\it{Predicted}$")

    ax.axvline(0.5, linestyle="--", linewidth=1.2)

    ys = np.r_[enc_y, act_y, pred_y]
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = max((ymax - ymin) * 0.12, 5.0)
    ax.set_ylim(ymin - pad * 0.25, ymax + pad)

    ax.annotate(r"$\it{Forecast\ Start}$", xy=(0.5, ymax), xytext=(4, 8),
                textcoords="offset points", fontsize=9)

    xticks = np.r_[enc_x, dec_x]
    xlabels = [f"t{i}" if i < 0 else "t" for i in enc_x] + [f"H+{i}" for i in dec_x]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_xlim(enc_x[0] - 0.5, dec_x[-1] + 0.5)

    ax.set_title(rf"$\it{{Sample\ Forecast\ ({label})}}$", fontsize=14)
    ax.set_xlabel(r"$\it{Time\ Step}$")
    ax.set_ylabel(r"$\it{Close\ Price}$")

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_attention_plot(model,label,raw_bundle,path):
    out=getattr(raw_bundle,"output",getattr(raw_bundle,"prediction",raw_bundle)); interp=model.interpret_output(out,reduction="sum"); att=_to_np(interp["attention"])
    while att.ndim>1: att=att.mean(axis=0)
    steps=np.arange(-len(att)+1,1); fig,ax=plt.subplots(figsize=(8.5,4.5)); ax.plot(steps,att,linewidth=1.8)
    ax.set_title(f"{it('Average Attention Pattern')} ({label})",fontsize=12); ax.set_xlabel(it("Encoder Steps")); ax.set_ylabel(it("Attention Weight")); ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7)); ax.yaxis.set_major_locator(MaxNLocator(nbins=6)); ax.grid(True,alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout(); fig.savefig(path,dpi=200,bbox_inches="tight"); plt.close(fig); return interp

def eval_model(label,ckpt_path,df_eval,batch_size):
    scenario,model=Path(ckpt_path).parent.name,load_model(ckpt_path); pred=predict_bundle(model,df_eval,batch_size=batch_size,mode="prediction"); g,h=evaluate_bundle(pred,diracc_eps=DIRACC_EPS)
    gdf=pd.DataFrame([{"model":label,"scenario":scenario,"split":EVAL_SPLIT,**g}]); h.insert(0,"split",EVAL_SPLIT); h.insert(0,"scenario",scenario); h.insert(0,"model",label)
    tg,th=eval_per_ticker(model,label,scenario,df_eval,batch_size,DIRACC_EPS); raw=predict_bundle(model,df_eval,batch_size=batch_size,mode="raw")
    return {"model":model,"global_df":gdf,"horizon_df":h,"ticker_df":tg,"ticker_h_df":th,"raw_bundle":raw}

def print_block(title,df): print("\n"+title); print(df.to_string(index=False))

def main():
    REPORT_DIR.mkdir(parents=True,exist_ok=True); FIG_DIR.mkdir(parents=True,exist_ok=True); _,val_df,test_df=split_df(pd.read_csv(DATA_PATH)); df_eval=val_df.copy() if EVAL_SPLIT=="val" else test_df.copy(); base=eval_model("TFT",BASELINE_CKPT,df_eval,BASE_BATCH_SIZE); hyb=eval_model("LLM-TFT",HYBRID_CKPT,df_eval,HYBRID_BATCH_SIZE)
    global_df=pd.concat([base["global_df"],hyb["global_df"]],ignore_index=True); horizon_df=pd.concat([base["horizon_df"],hyb["horizon_df"]],ignore_index=True); ticker_df=pd.concat([base["ticker_df"],hyb["ticker_df"]],ignore_index=True); ticker_h_df=pd.concat([base["ticker_h_df"],hyb["ticker_h_df"]],ignore_index=True)
    global_df[["model","scenario","split","rmse","mae","mape","r2","diracc","n"]].to_csv(REPORT_DIR/"eval_metrics_global.csv",index=False); horizon_df[["model","scenario","split","horizon","rmse","mae","mape","r2","diracc","n"]].to_csv(REPORT_DIR/"eval_metrics_by_horizon.csv",index=False); ticker_df.to_csv(REPORT_DIR/"eval_metrics_by_ticker_global.csv",index=False); ticker_h_df.to_csv(REPORT_DIR/"eval_metrics_by_ticker_horizon.csv",index=False)
    save_forecast_plot("TFT",base["raw_bundle"],FIG_DIR/"tft_sample_forecast.png",history_window=15); save_forecast_plot("LLM-TFT",hyb["raw_bundle"],FIG_DIR/"llmtft_sample_forecast.png",history_window=15); save_attention_plot(base["model"],"TFT",base["raw_bundle"],FIG_DIR/"tft_attention.png"); save_attention_plot(hyb["model"],"LLM-TFT",hyb["raw_bundle"],FIG_DIR/"llmtft_attention.png")
    s=_title_split(); print_block(f"METRIK GLOBAL PADA DATA {s}",global_df[["model","scenario","split","rmse","mae","mape","r2","diracc","n"]]); print_block(f"METRIK PER HORIZON PADA DATA {s}",horizon_df[["model","scenario","split","horizon","rmse","mae","mape","r2","diracc","n"]]); print_block(f"METRIK PER EMITEN PADA DATA {s}",ticker_df[["model","scenario","split","ticker","rmse","mae","mape","r2","diracc","n"]])

if __name__=="__main__": main()
