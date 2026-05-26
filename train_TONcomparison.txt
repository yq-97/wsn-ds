# -*- coding: utf-8 -*-
"""TON-IoT 实验二（Unknown / 拒识）—— 对照组 Baseline 预处理

你当前的实验设定（与你在 WSN-DS 实验二一致）：
  - Unknown 攻击：Injection
  - 数据集：你已经离线划分好的 train / val / test（CSV）
      * train、val：已删除 injection
      * test：包含 injection（全量）

脚本一次运行输出：
  1) Stage-1 二分类（Normal vs Attack）评估 + 2×2 混淆矩阵
  2) Stage-2 攻击多分类（只在 Known Attack 上训练/评估）
  3) Unknown(Injection) 拒识评估：
       - Injection 被判为 Normal / Unknown / 其他 Known 攻击 的数量与比例
       - Unknown Precision / Recall / F1（把 Unknown 视为“正类”）
       - Known-only Macro-F1 与 Coverage（只在未被拒识的 Known Attack 上算多分类）

注意：
  - 这里的“Unknown”不是新类别参与训练，而是通过 Stage-2 的置信度阈值（max softmax < tau）进行拒识。
  - tau 默认用 val(known attack) 自动标定：保持约 95% 的 known attack 不被拒识。

使用：
  1) 只改 CFG.train_csv / CFG.val_csv / CFG.test_csv 为你本地路径
  2) python train_TONcomparison_exp2_unknown_injection.py
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# 你的模型文件：优先使用你提供的 model_TONcomparison.py
from model_TONcomparison import AnomalyDetectionModel  # type: ignore
# =========================
# 0) 配置
# =========================
@dataclass
class CFG:
    # ====== 你的离线划分路径（你自己填） ======
    train_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\train_no_injection_with_origidx.csv"
    val_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\val_no_injection_with_origidx.csv"
    test_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\test_full_with_injection_with_origidx.csv"


    # ====== Unknown 设置 ======
    unknown_token: str = "injection"   # type 列里 Injection 的名字（大小写不敏感）

    # Feature encoding (baseline)
    hash_dim: int = 8
    drop_text_cols: bool = True

    # Windowing
    win: int = 64
    stride: int = 8
    window_label_mode: str = "last"  # "majority" / "last" / "any_attack"

    # Train
    seed: int = 42
    batch_size: int = 256
    epochs_bin: int = 20
    epochs_type: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model_dim: int = 128
    cross_attn_layers: int = 2
    tcn_layers: int = 2
    transformer_layers: int = 2
    nheads: int = 8
    dropout: float = 0.1

    # Stage-2
    init_from_stage1: bool = True

    # Decision thresholds
    attack_prob_thr: float = 0.5         # Stage-1：判 Attack 的阈值
    unknown_tau: Optional[float] = None  # Stage-2：拒识阈值（None=自动标定）
    known_keep_coverage: float = 0.90    # 自动标定 tau：尽量让 known attack 至少保留这么多


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) TON Baseline 预处理（只做特征，不掺杂标签）
# =========================
BOOLISH_COLS = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
    "weird_notice",
]

DASH_INT_COLS = [
    "http_trans_depth", "weird_addl",
]

LOW_CAT_COLS = [
    "proto", "service", "conn_state",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "ssl_version", "ssl_cipher",
    "http_method", "http_version",
    "weird_name",
]

HIGH_CAT_COLS = [
    "src_ip", "dst_ip",
    "dns_query",
    "http_uri",
    "http_user_agent",
    "ssl_subject",
    "ssl_issuer",
    "http_orig_mime_types",
    "http_resp_mime_types",
]


def _tf_to_int(s: pd.Series) -> pd.Series:
    return (
        s.fillna("-")
        .astype(str)
        .map({"T": 1, "F": 0, "-": 0})
        .fillna(0)
        .astype(np.int8)
    )


def _dashnum_to_int(s: pd.Series) -> pd.Series:
    return (
        s.fillna("-")
        .astype(str)
        .replace("-", "0")
        .astype(np.int32)
    )


def _hash_col(series: pd.Series, n_features: int, prefix: str) -> pd.DataFrame:
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    tokens = [[f"{prefix}={v}"] for v in series.fillna("missing").astype(str).tolist()]
    mat = hasher.transform(tokens).toarray().astype(np.float32)
    return pd.DataFrame(mat, columns=[f"{prefix}_h{i}" for i in range(n_features)])


def preprocess_ton_features(df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """返回：X_df、全部特征列名、连续数值列名（用于仅标准化这些列）"""
    df = df.copy()

    # 1) boolish / dash-int
    for c in BOOLISH_COLS:
        if c in df.columns:
            df[c] = _tf_to_int(df[c])
    for c in DASH_INT_COLS:
        if c in df.columns:
            df[c] = _dashnum_to_int(df[c])

    # 2) low-card one-hot
    low_cat = [c for c in LOW_CAT_COLS if c in df.columns]
    for c in low_cat:
        df[c] = df[c].fillna("missing").astype(str)
    df_low = pd.get_dummies(df[low_cat], prefix=low_cat, dtype=np.uint8) if low_cat else pd.DataFrame(index=df.index)

    # 3) high-card hashing
    high_cat = [c for c in HIGH_CAT_COLS if c in df.columns]
    if cfg.drop_text_cols:
        # 这些列极稀疏且对你模型不友好，默认去掉
        high_cat = [c for c in high_cat if c not in ["dns_query", "http_uri", "http_user_agent"]]
    df_hash_list = [_hash_col(df[c], cfg.hash_dim, c) for c in high_cat]
    df_hash = pd.concat(df_hash_list, axis=1) if df_hash_list else pd.DataFrame(index=df.index)

    # 4) continuous numeric
    ignore = set(low_cat + high_cat + ["label", "type"])
    cont_cols = [c for c in df.columns if c not in ignore]
    cont = df[cont_cols].copy()
    for c in cont.columns:
        cont[c] = pd.to_numeric(cont[c], errors="coerce")
    cont = cont.fillna(0.0).astype(np.float32)

    X_df = pd.concat([cont, df_low, df_hash], axis=1)
    return X_df, list(X_df.columns), list(cont.columns)


# =========================
# 2) 滑窗（同时产出 binary label + attack type label）
# =========================

def window_label_from_seg(y_seg: np.ndarray, mode: str, normal_id: int = 0) -> int:
    if mode == "last":
        return int(y_seg[-1])
    if mode == "any_attack":
        return int(1 if np.any(y_seg != normal_id) else normal_id)
    # majority
    vals, cnt = np.unique(y_seg, return_counts=True)
    return int(vals[np.argmax(cnt)])


def make_windows_two_labels(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_type_str: np.ndarray,
    cfg: CFG,
    attack_type_to_idx: Dict[str, int],
    normal_token: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回：Xw [Nw,W,D], yb_w [Nw], yt_w [Nw](known type idx or -1), tlast_w[str]

    说明：
      - yt_w=-1 表示：非攻击窗口 or Unknown 攻击（如 injection）
      - tlast_w 保存 last-step 的 type 字符串，用于统计 Injection→XXX
    """
    N, D = X.shape
    if N < cfg.win:
        raise ValueError(f"Too few rows ({N}) for win={cfg.win}")

    xs, ybs, yts, tlasts = [], [], [], []
    for i in range(0, N - cfg.win + 1, cfg.stride):
        seg_bin = y_bin[i:i + cfg.win]
        yb = window_label_from_seg(seg_bin, cfg.window_label_mode, normal_id=0)

        t_last = str(y_type_str[i + cfg.win - 1])
        tlasts.append(t_last)

        if yb == 1:
            if t_last.lower() == normal_token:
                yt = -1
            else:
                yt = int(attack_type_to_idx.get(t_last, -1))
        else:
            yt = -1

        xs.append(X[i:i + cfg.win])
        ybs.append(yb)
        yts.append(yt)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ybs, dtype=np.int64),
        np.asarray(yts, dtype=np.int64),
        np.asarray(tlasts, dtype=object),
    )


class WindowDataset(Dataset):
    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        self.Xw = torch.from_numpy(Xw)
        self.yw = torch.from_numpy(yw)

    def __len__(self):
        return self.Xw.shape[0]

    def __getitem__(self, idx):
        return self.Xw[idx], self.yw[idx]


# =========================
# 3) 评估（Stage-1 二分类）
# =========================
@torch.no_grad()
def predict_binary(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, probs, preds = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob_attack = torch.softmax(logits, dim=1)[:, 1]
        ys.append(yb.numpy())
        probs.append(prob_attack.detach().cpu().numpy())
        preds.append((prob_attack >= 0.5).long().cpu().numpy())
    y = np.concatenate(ys).astype(np.int64)
    prob = np.concatenate(probs).astype(np.float32)
    pred = np.concatenate(preds).astype(np.int64)
    return y, pred, prob


def eval_binary_from_pred(y: np.ndarray, p: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out = {
        "acc": float(accuracy_score(y, p)),
        "p": float(precision_score(y, p, zero_division=0)),
        "r": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    try:
        out["auroc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auroc"] = float("nan")
    return out


# =========================
# 4) 评估（Stage-2 多分类 + Unknown 拒识）
# =========================
@torch.no_grad()
def predict_multiclass(model: nn.Module, Xw: np.ndarray, device: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回：maxprob, pred_class（argmax）"""
    model.eval()
    maxps, preds = [], []
    for i in range(0, len(Xw), batch_size):
        xb = torch.from_numpy(Xw[i:i + batch_size]).to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        maxp, pred = prob.max(dim=1)
        maxps.append(maxp.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(maxps), np.concatenate(preds)


def compute_unknown_tau_from_val(maxp_val: np.ndarray, keep_coverage: float) -> float:
    """让 known attack 大约 keep_coverage 的窗口不被拒识：tau = quantile(maxp, 1-keep)."""
    keep_coverage = float(np.clip(keep_coverage, 0.5, 0.999))
    q = 1.0 - keep_coverage
    return float(np.quantile(maxp_val, q))


def balanced_weights(y: np.ndarray, K: int) -> np.ndarray:
    """balanced 权重：w_c = N/(K*count_c)，缺失类 weight=1"""
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=K).astype(np.float64)
    N = counts.sum()
    w = np.ones(K, dtype=np.float32)
    for c in range(K):
        if counts[c] > 0:
            w[c] = float(N / (K * counts[c]))
    return w


# =========================
# 5) 训练
# =========================

def build_model(cfg: CFG, feature_dim: int, num_classes: int) -> nn.Module:
    return AnomalyDetectionModel(
        feature_dim=feature_dim,
        num_classes=num_classes,
        cross_attn_layers=cfg.cross_attn_layers,
        model_dim=cfg.model_dim,
        tcn_layers=cfg.tcn_layers,
        transformer_layers=cfg.transformer_layers,
        nheads=cfg.nheads,
        dropout=cfg.dropout,
        max_len=cfg.win,
    )


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
    epochs: int,
    criterion: nn.Module,
    eval_fn,
    best_path: str,
    best_key: str,
):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        avg_loss = total / max(1, len(train_loader))
        val = eval_fn(model, val_loader)
        score = float(val[best_key])
        print(f"Epoch {ep:02d} | loss={avg_loss:.4f} | VAL {best_key}={score:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=cfg.device,weights_only=True))


def main():
    cfg = CFG()
    seed_all(cfg.seed)
    assert cfg.window_label_mode in ("majority", "last", "any_attack")

    print("🚀 Load offline split CSVs")
    print("  train:", cfg.train_csv)
    print("  val  :", cfg.val_csv)
    print("  test :", cfg.test_csv)

    df_tr = pd.read_csv(cfg.train_csv, low_memory=False)
    df_va = pd.read_csv(cfg.val_csv, low_memory=False)
    df_te = pd.read_csv(cfg.test_csv, low_memory=False)

    # ===== 必须：按 orig_idx 恢复原始顺序（滑窗前）=====
    for name, d in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        if "orig_idx" not in d.columns:
            raise ValueError(f"{name} 缺少 orig_idx 列，请使用 *_with_origidx.csv")
        d.sort_values("orig_idx", inplace=True)
        d.reset_index(drop=True, inplace=True)

    for name, df in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        if "label" not in df.columns or "type" not in df.columns:
            raise ValueError(f"{name} 缺少 label/type 列，请检查数据文件")

    print(f"Rows: train={len(df_tr)} val={len(df_va)} test={len(df_te)}")

    # ===== Attack types：只用 train+val（因为 injection 已从这里删除） =====
    type_trva = pd.concat([df_tr[["type"]], df_va[["type"]]], axis=0)["type"].fillna("missing").astype(str).to_numpy()
    is_normal = np.char.lower(type_trva.astype(str)) == "normal"
    attack_type_set = sorted(set(type_trva[~is_normal].tolist()))

    # 如果你担心 injection 没删干净，这里强制移除一次
    attack_type_set = [t for t in attack_type_set if t.lower() != cfg.unknown_token.lower()]

    attack_type_to_idx = {t: i for i, t in enumerate(attack_type_set)}
    print(f"[Known attack types] K={len(attack_type_set)} (exclude 'normal' and unknown='{cfg.unknown_token}')")

    # ===== Features：为保证 one-hot 列一致，这里对 train+val+test 一起做特征展开（不使用标签） =====
    df_all = pd.concat([df_tr, df_va, df_te], axis=0, ignore_index=True)

    X_df, feature_cols, cont_cols = preprocess_ton_features(df_all, cfg)
    X_all = X_df.to_numpy().astype(np.float32)

    n_tr = len(df_tr)
    n_va = len(df_va)
    idx_tr = np.arange(0, n_tr)
    idx_va = np.arange(n_tr, n_tr + n_va)
    idx_te = np.arange(n_tr + n_va, n_tr + n_va + len(df_te))

    # ===== 只用 train 拟合 scaler（仅连续列） =====
    cont_idx = [feature_cols.index(c) for c in cont_cols]
    scaler = StandardScaler()
    scaler.fit(X_all[idx_tr][:, cont_idx])
    X_all[:, cont_idx] = scaler.transform(X_all[:, cont_idx]).astype(np.float32)

    # ===== labels =====
    y_bin_all = df_all["label"].astype(int).to_numpy()
    type_all = df_all["type"].fillna("missing").astype(str).to_numpy()

    # ===== windows =====
    Xw_tr, yb_tr, yt_tr, tlast_tr = make_windows_two_labels(X_all[idx_tr], y_bin_all[idx_tr], type_all[idx_tr], cfg, attack_type_to_idx)
    Xw_va, yb_va, yt_va, tlast_va = make_windows_two_labels(X_all[idx_va], y_bin_all[idx_va], type_all[idx_va], cfg, attack_type_to_idx)
    Xw_te, yb_te, yt_te, tlast_te = make_windows_two_labels(X_all[idx_te], y_bin_all[idx_te], type_all[idx_te], cfg, attack_type_to_idx)

    print(f"[Features] dim={len(feature_cols)} | cont={len(cont_cols)} | hash_dim={cfg.hash_dim} | drop_text={cfg.drop_text_cols}")
    print(f"[Window] win={cfg.win} stride={cfg.stride} label_mode={cfg.window_label_mode}")
    print(f"[Windows] train={Xw_tr.shape} val={Xw_va.shape} test={Xw_te.shape}")
    print("[Binary window dist][train]", {int(k): int(v) for k, v in zip(*np.unique(yb_tr, return_counts=True))})

    # =========================
    # Stage-1: Binary Detector
    # =========================
    feature_dim = Xw_tr.shape[2]
    model_bin = build_model(cfg, feature_dim, num_classes=2).to(cfg.device)
    print("[Stage-1] Model params:", sum(p.numel() for p in model_bin.parameters()))

    w_bin = balanced_weights(yb_tr, K=2)
    crit_bin = nn.CrossEntropyLoss(weight=torch.tensor(w_bin, dtype=torch.float32, device=cfg.device))

    dl_tr_bin = DataLoader(WindowDataset(Xw_tr, yb_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_va_bin = DataLoader(WindowDataset(Xw_va, yb_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_te_bin = DataLoader(WindowDataset(Xw_te, yb_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    best_bin = "best_ton_exp2_stage1_binary.pt"
    train_loop(
        model_bin,
        dl_tr_bin,
        dl_va_bin,
        cfg,
        epochs=cfg.epochs_bin,
        criterion=crit_bin,
        eval_fn=lambda m, l: eval_binary_from_pred(*predict_binary(m, l, cfg.device)),
        best_path=best_bin,
        best_key="f1",
    )

    # test stage-1
    y_te_bin, prob_te_attack, pred_te_attack_05 = predict_binary(model_bin, dl_te_bin, cfg.device)
    pred_te_attack = (prob_te_attack >= cfg.attack_prob_thr).astype(np.int64)
    te_bin = eval_binary_from_pred(y_te_bin, pred_te_attack, prob_te_attack)
    print("\n[TEST][Stage-1 Binary]")
    print(te_bin)
    print(f"ConfusionMatrix [[TN FP],[FN TP]] = [[{te_bin['tn']} {te_bin['fp']}],[{te_bin['fn']} {te_bin['tp']}]]")
    print(f"[Stage-1] attack_prob_thr={cfg.attack_prob_thr:.2f}")

    # =========================
    # Stage-2: Known Attack Type Classifier
    # =========================
    def filter_known_attack(Xw, yt, yb, tlast):
        m = (yb == 1) & (yt >= 0)
        return Xw[m], yt[m], tlast[m]

    Xw_tr_k, yt_tr_k, _ = filter_known_attack(Xw_tr, yt_tr, yb_tr, tlast_tr)
    Xw_va_k, yt_va_k, _ = filter_known_attack(Xw_va, yt_va, yb_va, tlast_va)
    Xw_te_k, yt_te_k, _ = filter_known_attack(Xw_te, yt_te, yb_te, tlast_te)

    K_attack = len(attack_type_set)
    print(f"\n[Stage-2] Known-Attack windows only: train={len(yt_tr_k)} val={len(yt_va_k)} test={len(yt_te_k)} | K={K_attack}")

    if len(yt_tr_k) == 0 or K_attack == 0:
        print("[WARN] No known attack windows or no known attack types found. Stage-2 skipped.")
        return

    model_type = build_model(cfg, feature_dim, num_classes=K_attack).to(cfg.device)

    # init from stage-1 (skip fc)
    if cfg.init_from_stage1:
        sd = torch.load(best_bin, map_location="cpu",weights_only=True)
        for k in list(sd.keys()):
            if k.startswith("fc."):
                sd.pop(k)
        missing, unexpected = model_type.load_state_dict(sd, strict=False)
        print("[Stage-2] init_from_stage1=True | missing:", len(missing), "unexpected:", len(unexpected))

    w_type = balanced_weights(yt_tr_k, K=K_attack)
    crit_type = nn.CrossEntropyLoss(weight=torch.tensor(w_type, dtype=torch.float32, device=cfg.device))

    dl_tr_type = DataLoader(WindowDataset(Xw_tr_k, yt_tr_k), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_va_type = DataLoader(WindowDataset(Xw_va_k, yt_va_k), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    def eval_known_type(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        ys, ps = [], []
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            ys.append(yb.numpy())
            ps.append(pred.cpu().numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        out = {
            "acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(K_attack)), zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", labels=list(range(K_attack)), zero_division=0)),
        }
        return out

    best_type = "best_ton_exp2_stage2_known_attack_types.pt"
    train_loop(
        model_type,
        dl_tr_type,
        dl_va_type,
        cfg,
        epochs=cfg.epochs_type,
        criterion=crit_type,
        eval_fn=lambda m, l: eval_known_type(m, l),
        best_path=best_type,
        best_key="macro_f1",
    )

    # ========== Stage-2: 在 val 上自动标定 unknown tau ==========
    maxp_va, pred_va = predict_multiclass(model_type, Xw_va_k, cfg.device, cfg.batch_size)
    if cfg.unknown_tau is None:
        tau = compute_unknown_tau_from_val(maxp_va, keep_coverage=cfg.known_keep_coverage)
        print(f"\n[Unknown tau] auto from val | keep_coverage={cfg.known_keep_coverage:.3f} => tau={tau:.4f}")
    else:
        tau = float(cfg.unknown_tau)
        print(f"\n[Unknown tau] manual => tau={tau:.4f}")

    # ========== Stage-2: Known-only test（不含 injection） ==========
    maxp_te_k, pred_te_k = predict_multiclass(model_type, Xw_te_k, cfg.device, cfg.batch_size)
    print("\n[TEST][Stage-2 Known-Attack Types | Known attack windows only]")
    print("macro_f1=%.4f weighted_f1=%.4f" % (
        f1_score(yt_te_k, pred_te_k, average="macro", labels=list(range(K_attack)), zero_division=0),
        f1_score(yt_te_k, pred_te_k, average="weighted", labels=list(range(K_attack)), zero_division=0),
    ))
    try:
        print("\n[Attack-Type ClassificationReport][Known attack windows only]")
        print(classification_report(yt_te_k, pred_te_k, labels=list(range(K_attack)), target_names=attack_type_set, zero_division=0, digits=4))
    except Exception:
        pass

    # =========================
    # 6) Unknown / 拒识总评估（在完整 test 上）
    # =========================
    # Ground-truth injection windows（以 last-step 的 type 为准）
    inj_mask = (yb_te == 1) & (np.char.lower(tlast_te.astype(str)) == cfg.unknown_token.lower())

    # Stage-2 只对“被 Stage-1 判为 Attack”的窗口做拒识/多分类
    attack_pred_mask = pred_te_attack.astype(bool)

    # 对 attack_pred 的窗口跑 Stage-2
    Xw_te_for_stage2 = Xw_te[attack_pred_mask]
    if len(Xw_te_for_stage2) > 0:
        maxp_all, pred_all = predict_multiclass(model_type, Xw_te_for_stage2, cfg.device, cfg.batch_size)
    else:
        maxp_all = np.array([], dtype=np.float32)
        pred_all = np.array([], dtype=np.int64)

    # 把 stage2 输出回填到全量窗口（未进入 stage2 的设为 -inf / -1）
    maxp_full = np.full(len(Xw_te), -1.0, dtype=np.float32)
    pred_full = np.full(len(Xw_te), -1, dtype=np.int64)
    maxp_full[attack_pred_mask] = maxp_all
    pred_full[attack_pred_mask] = pred_all

    pred_is_unknown = attack_pred_mask & (maxp_full >= 0) & (maxp_full < tau)

    # 统计 Injection 的去向（Normal / Unknown / KnownAttack）
    inj_total = int(inj_mask.sum())
    inj_to_normal = int((inj_mask & ~attack_pred_mask).sum())
    inj_to_unknown = int((inj_mask & pred_is_unknown).sum())
    inj_to_known = int((inj_mask & attack_pred_mask & ~pred_is_unknown).sum())

    print("\n[Unknown Eval][Injection as Unknown]")
    if inj_total == 0:
        print("[WARN] test 中没有 injection 窗口（按 last-step type 统计），请确认 test.csv 是否包含 injection")
    else:
        print(f"Injection windows total: {inj_total}")
        print(f"  Injection→Normal : {inj_to_normal} ({inj_to_normal / inj_total:.2%})")
        print(f"  Injection→Unknown: {inj_to_unknown} ({inj_to_unknown / inj_total:.2%})")
        print(f"  Injection→Known  : {inj_to_known} ({inj_to_known / inj_total:.2%})")

    # Unknown 作为“正类”的 P/R/F1（在全量窗口上）
    y_unknown_true = inj_mask.astype(np.int64)
    y_unknown_pred = pred_is_unknown.astype(np.int64)
    unk_p = precision_score(y_unknown_true, y_unknown_pred, zero_division=0)
    unk_r = recall_score(y_unknown_true, y_unknown_pred, zero_division=0)
    unk_f1 = f1_score(y_unknown_true, y_unknown_pred, zero_division=0)
    # 一个可用的 unknown score：-maxp（maxp 越低越像 unknown），未进入 stage2 的记为 +inf（更不像 unknown）
    score_unknown = -maxp_full
    score_unknown[~attack_pred_mask] = -0.0
    try:
        unk_auroc = roc_auc_score(y_unknown_true, score_unknown)
    except Exception:
        unk_auroc = float('nan')

    print(f"[Unknown] P={unk_p:.4f} R={unk_r:.4f} F1={unk_f1:.4f} AUROC(score=-maxp)={unk_auroc:.4f}")

    # Known-only 多分类：只在 ground-truth known attack 且最终没被拒识/没被判 normal 的窗口上算
    known_gt_mask = (yb_te == 1) & (yt_te >= 0)
    known_keep_mask = known_gt_mask & attack_pred_mask & (~pred_is_unknown)
    coverage = float(known_keep_mask.sum() / max(1, known_gt_mask.sum()))

    if known_keep_mask.sum() > 0:
        y_true_known = yt_te[known_keep_mask]
        y_pred_known = pred_full[known_keep_mask]
        known_macro_f1 = f1_score(y_true_known, y_pred_known, average="macro", labels=list(range(K_attack)), zero_division=0)
    else:
        known_macro_f1 = 0.0

    print(f"[Known-only] Macro-F1={known_macro_f1:.4f} | coverage={coverage:.4f}")

    # 给你一个更直观的：Normal 被误判为 Attack 的比例
    normal_gt_mask = (yb_te == 0)
    fp = int((normal_gt_mask & attack_pred_mask).sum())
    tn = int((normal_gt_mask & ~attack_pred_mask).sum())
    print(f"[Stage-1 FP] Normal→Attack: {fp} / {fp + tn} ({fp / max(1, fp + tn):.2%})")

    print(f"\n[OK] Saved: {best_bin} | {best_type}")


if __name__ == "__main__":
    main()
