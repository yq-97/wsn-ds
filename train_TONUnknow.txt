# -*- coding: utf-8 -*-
"""TON-IoT 实验二（Unknown / 拒识）—— 主实验 KMS + PCA 预处理

你的实验设定：
  - Unknown 攻击：Injection
  - 你已经离线划分：train / val / test（CSV）
      * train、val：已删除 injection
      * test：包含 injection（全量）
  - 预处理：baseline 特征工程 + StandardScaler(仅连续特征) + PCA + Stage-2 训练集 KMeansSMOTE

脚本一次运行输出：
  1) Stage-1 二分类（Normal vs Attack）评估 + 2×2 混淆矩阵
  2) Stage-2 攻击多分类（只在 Known Attack 上训练/评估）
  3) Unknown(Injection) 拒识评估：
       - Injection 被判为 Normal / Unknown / 其他 Known 攻击 的数量与比例
       - Unknown Precision / Recall / F1（把 Unknown 视为“正类”）
       - Known-only Macro-F1 与 Coverage（只在未被拒识的 Known Attack 上算多分类）

使用：
  1) 改 CFG.train_csv / CFG.val_csv / CFG.test_csv 为你本地路径
  2) 确保同目录存在 model_TONKMS.py（或把 import 改成你的模型文件名）
  3) python train_TONKMS_exp2_unknown_injection.py
"""

import os
import warnings

# Windows 下 MiniBatchKMeans 相关警告 + 线程数
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
warnings.filterwarnings('ignore', message='.*MiniBatchKMeans is known to have a memory leak.*')

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# KMS
try:
    from imblearn.over_sampling import KMeansSMOTE
except Exception as e:
    raise ImportError(
        "缺少 imbalanced-learn（KMeansSMOTE）。请先安装：pip install imbalanced-learn"
    ) from e

# 你的模型文件
from model_TONUnknow import AnomalyDetectionModel  # type: ignore


def safe_torch_load_state_dict(path: str, map_location):
    """avoid torch.load FutureWarning across torch versions"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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

    # Feature encoding
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

    # PCA
    pca_n_components: float | int = 0.95 # 例如 16；也可写 0.95

    # KMS (Stage-2)
    kms_k_neighbors: int = 3
    kms_random_state: int = 42
    kms_cluster_balance_threshold: float = 0.01  # 越小越容易让 KMS 成功（避免 'No clusters found...'）
    kms_n_clusters: int = 8                 # KMeans 聚类数；不宜过大
    kms_kmeans_batch_size: int = 1024       # MiniBatchKMeans batch_size

    # Stage-2
    init_from_stage1: bool = True

    # Decision thresholds
    attack_prob_thr: float = 0.5
    unknown_tau: Optional[float] = None
    known_keep_coverage: float = 0.90


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) TON Baseline 特征工程
# =========================
BOOLISH_COLS = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
    "weird_notice",
]

DASH_INT_COLS = [
    "dns_qclass", "dns_qtype", "dns_rcode",
    "ssl_version", "ssl_cipher", "ssl_curve",
]

DROP_COLS_ALWAYS = [
    "src_ip", "dst_ip",
    "ts", "timestamp", "time",
    "conn_state",
]

LABEL_COLS = ["label", "type"]


def _to_boolish(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n"}
    out = s.map(lambda x: 1 if x in true_set else (0 if x in false_set else np.nan))
    return out.fillna(0).astype(np.float32)


def _dash_to_int(s: pd.Series) -> pd.Series:
    s = s.replace("-", np.nan)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float32)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def preprocess_ton_features(df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """baseline 特征工程：
    - 低基数类别：get_dummies
    - 高基数文本：FeatureHasher
    - 连续数值：保留 + 后续 StandardScaler(仅 train fit)

    返回：X_df, feature_cols, cont_cols
    """
    df = df.copy()

    for c in BOOLISH_COLS:
        if c in df.columns:
            df[c] = _to_boolish(df[c])

    for c in DASH_INT_COLS:
        if c in df.columns:
            df[c] = _dash_to_int(df[c])

    for c in DROP_COLS_ALWAYS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    if cfg.drop_text_cols:
        for c in list(df.columns):
            if c in LABEL_COLS:
                continue
            if df[c].dtype == object and c not in ("proto", "service"):
                if c in df.columns:
                    df.drop(columns=[c], inplace=True)

    y_cols = [c for c in LABEL_COLS if c in df.columns]
    feat_df = df.drop(columns=y_cols, errors="ignore")

    cat_cols = [c for c in feat_df.columns if feat_df[c].dtype == object]
    low_card_cols = []
    high_card_cols = []
    for c in cat_cols:
        nunique = feat_df[c].nunique(dropna=True)
        if nunique <= 30:
            low_card_cols.append(c)
        else:
            high_card_cols.append(c)

    cont_cols = [c for c in feat_df.columns if c not in cat_cols]
    for c in cont_cols:
        feat_df[c] = _safe_numeric(feat_df[c])

    X_parts = []

    if low_card_cols:
        dummies = pd.get_dummies(feat_df[low_card_cols].fillna("missing"), prefix=low_card_cols, dtype=np.float32)
        X_parts.append(dummies)

    if high_card_cols:
        hasher = FeatureHasher(n_features=cfg.hash_dim, input_type="string")
        tokens = feat_df[high_card_cols].fillna("missing").astype(str).agg("|".join, axis=1)
        X_hash = hasher.transform(tokens)
        X_hash = X_hash.toarray().astype(np.float32)
        X_hash_df = pd.DataFrame(X_hash, columns=[f"hash_{i}" for i in range(X_hash.shape[1])])
        X_parts.append(X_hash_df)

    if cont_cols:
        X_parts.append(feat_df[cont_cols].astype(np.float32))

    if not X_parts:
        raise ValueError("No features left after preprocessing.")

    X_df = pd.concat(X_parts, axis=1)
    feature_cols = list(X_df.columns)
    return X_df, feature_cols, cont_cols


# =========================
# 2) Windowing（同时生成二分类 + type label + last_type_str）
# =========================

def window_label_from_seg(seg_bin: np.ndarray, mode: str, normal_id: int = 0) -> int:
    if mode == "last":
        return int(seg_bin[-1])
    if mode == "any_attack":
        return 1 if np.any(seg_bin != normal_id) else 0
    counts = np.bincount(seg_bin.astype(np.int64), minlength=2)
    return int(np.argmax(counts))


def make_windows_two_labels(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_type_str: np.ndarray,
    cfg: CFG,
    attack_type_to_idx: Dict[str, int],
    normal_token: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, D = X.shape
    if N < cfg.win:
        raise ValueError(f"Too few rows ({N}) for win={cfg.win}")

    xs, ybs, yts, tlasts = [], [], [], []
    for i in range(0, N - cfg.win + 1, cfg.stride):
        seg_bin = y_bin[i:i + cfg.win]
        yb = window_label_from_seg(seg_bin, cfg.window_label_mode, normal_id=0)
        t_last = str(y_type_str[i + cfg.win - 1])
        t_low = t_last.lower()

        if yb == 1:
            if t_low == normal_token:
                yt = -1
            else:
                yt = int(attack_type_to_idx.get(t_last, -1))  # unknown => -1
        else:
            yt = -1

        xs.append(X[i:i + cfg.win])
        ybs.append(yb)
        yts.append(yt)
        tlasts.append(t_low)

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
# 3) 评估工具
# =========================
@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader, device: str, prob_thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob >= prob_thr).long()
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pr = np.concatenate(probs)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out = {
        "acc": float(accuracy_score(y, p)),
        "p": float(precision_score(y, p, zero_division=0)),
        "r": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "thr": float(prob_thr),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, pr))
    except Exception:
        out["auroc"] = float("nan")
    return out


@torch.no_grad()
def predict_binary_prob(model: nn.Module, Xw: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    probs = []
    dl = DataLoader(WindowDataset(Xw, np.zeros((len(Xw),), dtype=np.int64)), batch_size=batch_size, shuffle=False, num_workers=0)
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        probs.append(prob.detach().cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def predict_type_maxprob_and_pred(model: nn.Module, Xw: np.ndarray, device: str, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    maxps, preds = [], []
    dl = DataLoader(WindowDataset(Xw, np.zeros((len(Xw),), dtype=np.int64)), batch_size=batch_size, shuffle=False, num_workers=0)
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        mx, pr = prob.max(dim=1)
        maxps.append(mx.detach().cpu().numpy())
        preds.append(pr.detach().cpu().numpy())
    return np.concatenate(maxps), np.concatenate(preds)


def balanced_weights(y: np.ndarray, K: int) -> np.ndarray:
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=K).astype(np.float64)
    N = counts.sum()
    w = np.ones(K, dtype=np.float32)
    for c in range(K):
        if counts[c] > 0:
            w[c] = float(N / (K * counts[c]))
    return w


# =========================
# 4) 模型与训练
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
    model.load_state_dict(safe_torch_load_state_dict(best_path, map_location=cfg.device))


# =========================
# 5) 主流程
# =========================

def main():
    cfg = CFG()
    seed_all(cfg.seed)

    print("\n========== EXP2 (TON-IoT) | KMS+PCA | Unknown=Injection ==========")
    print("[CSV paths]")
    print("  train:", cfg.train_csv)
    print("  val  :", cfg.val_csv)
    print("  test :", cfg.test_csv)

    # ---------- load splits ----------
    df_tr = pd.read_csv(cfg.train_csv, low_memory=False)
    df_va = pd.read_csv(cfg.val_csv, low_memory=False)
    df_te = pd.read_csv(cfg.test_csv, low_memory=False)

    # ===== 必须：按 orig_idx 恢复原始顺序（滑窗前）=====
    for name, d in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        if "orig_idx" not in d.columns:
            raise ValueError(f"{name} 缺少 orig_idx 列，请使用 *_with_origidx.csv")
        d.sort_values("orig_idx", inplace=True)
        d.reset_index(drop=True, inplace=True)

    # labels
    y_tr_bin = df_tr["label"].astype(int).to_numpy()
    y_va_bin = df_va["label"].astype(int).to_numpy()
    y_te_bin = df_te["label"].astype(int).to_numpy()

    type_tr = df_tr["type"].fillna("missing").astype(str).to_numpy()
    type_va = df_va["type"].fillna("missing").astype(str).to_numpy()
    type_te = df_te["type"].fillna("missing").astype(str).to_numpy()

    # known attack types from train+val (exclude normal)
    def is_normal(arr: np.ndarray) -> np.ndarray:
        return np.char.lower(arr.astype(str)) == "normal"

    known_attack_types = sorted(set(np.concatenate([type_tr[~is_normal(type_tr)], type_va[~is_normal(type_va)]], axis=0).tolist()))
    attack_type_to_idx = {t: i for i, t in enumerate(known_attack_types)}
    K_attack = len(known_attack_types)
    print(f"[Known attack types] K={K_attack} (train+val) | injection should be absent here")

    # ---------- feature engineering (concat to align columns) ----------
    df_all = pd.concat([df_tr, df_va, df_te], axis=0, ignore_index=True)
    X_all_df, feature_cols, cont_cols = preprocess_ton_features(df_all, cfg)

    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)
    X_tr_df = X_all_df.iloc[:n_tr]
    X_va_df = X_all_df.iloc[n_tr:n_tr + n_va]
    X_te_df = X_all_df.iloc[n_tr + n_va:]

    X_tr = X_tr_df.to_numpy().astype(np.float32)
    X_va = X_va_df.to_numpy().astype(np.float32)
    X_te = X_te_df.to_numpy().astype(np.float32)

    # ---------- scale continuous columns (fit on train) ----------
    cont_idx = [feature_cols.index(c) for c in cont_cols] if cont_cols else []
    if cont_idx:
        scaler = StandardScaler()
        scaler.fit(X_tr[:, cont_idx])
        X_tr[:, cont_idx] = scaler.transform(X_tr[:, cont_idx]).astype(np.float32)
        X_va[:, cont_idx] = scaler.transform(X_va[:, cont_idx]).astype(np.float32)
        X_te[:, cont_idx] = scaler.transform(X_te[:, cont_idx]).astype(np.float32)

    # ---------- PCA (fit on train rows; transform train/val/test) ----------
    pca = PCA(n_components=cfg.pca_n_components, random_state=cfg.seed)
    pca.fit(X_tr)
    X_tr = pca.transform(X_tr).astype(np.float32)
    X_va = pca.transform(X_va).astype(np.float32)
    X_te = pca.transform(X_te).astype(np.float32)
    print(f"[PCA] {X_tr_df.shape[1]} -> {X_tr.shape[1]} | explained_var={float(np.sum(pca.explained_variance_ratio_)):.4f}")

    # ---------- windowing per split ----------
    Xw_tr, yb_tr, yt_tr, tlast_tr = make_windows_two_labels(X_tr, y_tr_bin, type_tr, cfg, attack_type_to_idx)
    Xw_va, yb_va, yt_va, tlast_va = make_windows_two_labels(X_va, y_va_bin, type_va, cfg, attack_type_to_idx)
    Xw_te, yb_te, yt_te, tlast_te = make_windows_two_labels(X_te, y_te_bin, type_te, cfg, attack_type_to_idx)

    feature_dim = Xw_tr.shape[2]
    print(f"[Windows] train={Xw_tr.shape} val={Xw_va.shape} test={Xw_te.shape} | feat_dim={feature_dim}")

    # =========================
    # Stage-1: Binary Detector
    # =========================
    model_bin = build_model(cfg, feature_dim, num_classes=2).to(cfg.device)
    w_bin = balanced_weights(yb_tr, K=2)
    crit_bin = nn.CrossEntropyLoss(weight=torch.tensor(w_bin, dtype=torch.float32, device=cfg.device))

    dl_tr_bin = DataLoader(WindowDataset(Xw_tr, yb_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_va_bin = DataLoader(WindowDataset(Xw_va, yb_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_te_bin = DataLoader(WindowDataset(Xw_te, yb_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    best_bin = "best_ton_exp2_kms_stage1_binary.pt"
    train_loop(
        model_bin, dl_tr_bin, dl_va_bin, cfg,
        epochs=cfg.epochs_bin,
        criterion=crit_bin,
        eval_fn=lambda m, l: eval_binary(m, l, cfg.device, prob_thr=cfg.attack_prob_thr),
        best_path=best_bin,
        best_key="f1",
    )

    te_bin = eval_binary(model_bin, dl_te_bin, cfg.device, prob_thr=cfg.attack_prob_thr)
    print("\n[TEST][Stage-1 Binary]")
    print(te_bin)
    print(f"ConfusionMatrix [[TN FP],[FN TP]] = [[{te_bin['tn']} {te_bin['fp']}],[{te_bin['fn']} {te_bin['tp']}]]")

    # =========================
    # Stage-2: Attack Type Classifier (Known attack windows only) + KMS
    # =========================
    def filter_known_attack(Xw, yt, yb):
        m = (yb == 1) & (yt >= 0)
        return Xw[m], yt[m]

    Xw_tr_a, yt_tr_a = filter_known_attack(Xw_tr, yt_tr, yb_tr)
    Xw_va_a, yt_va_a = filter_known_attack(Xw_va, yt_va, yb_va)
    Xw_te_known, yt_te_known = filter_known_attack(Xw_te, yt_te, yb_te)

    print(f"\n[Stage-2] Known-attack windows: train={len(yt_tr_a)} val={len(yt_va_a)} test_known={len(yt_te_known)} | K={K_attack}")

    if len(yt_tr_a) == 0 or K_attack == 0:
        print("[WARN] No known attack windows or no known attack types. Stage-2 skipped.")
        return

    # --- KMS on training attack windows (flatten -> resample -> reshape) ---
    cnt_before = np.bincount(yt_tr_a, minlength=K_attack)
    min_cnt = int(np.min(cnt_before[cnt_before > 0])) if np.any(cnt_before > 0) else 0
    k = min(cfg.kms_k_neighbors, max(1, min_cnt - 1))

    if min_cnt <= 1 or k < 1:
        print(f"[KMS] min class too small (min_cnt={min_cnt}). Skip KMeansSMOTE.")
        Xw_tr_res, yt_tr_res = Xw_tr_a, yt_tr_a
    else:
        X_flat = Xw_tr_a.reshape(Xw_tr_a.shape[0], -1)

        # KMeansSMOTE 在少样本/簇不平衡时常见报错：
        # "No clusters found with sufficient samples..."
        # 处理思路：调低 cluster_balance_threshold，并适度增加 n_clusters（但别太大）。
        tried = []
        last_err = None
        success = False

        # 只在这里动参数，不改你其它训练逻辑
        thr_list = [
            float(cfg.kms_cluster_balance_threshold),
            float(max(1e-4, cfg.kms_cluster_balance_threshold * 0.5)),
            float(max(1e-4, cfg.kms_cluster_balance_threshold * 0.2)),
        ]
        nc_base = int(max(2, cfg.kms_n_clusters))
        nc_list = [nc_base, int(min(24, nc_base * 2))]

        for thr in thr_list:
            for nc in nc_list:
                try:
                    kmeans = MiniBatchKMeans(
                        n_clusters=nc,
                        random_state=int(cfg.kms_random_state),
                        batch_size=int(cfg.kms_kmeans_batch_size),
                        n_init=10,
                    )
                    kms = KMeansSMOTE(
                        random_state=int(cfg.kms_random_state),
                        k_neighbors=int(k),
                        cluster_balance_threshold=float(thr),
                        kmeans_estimator=kmeans,
                    )
                    X_res, y_res = kms.fit_resample(X_flat, yt_tr_a)
                    Xw_tr_res = X_res.reshape(-1, cfg.win, feature_dim).astype(np.float32)
                    yt_tr_res = y_res.astype(np.int64)
                    print(f"[KMS] SUCCESS | k={k} thr={thr:g} n_clusters={nc} | {Xw_tr_a.shape[0]} -> {Xw_tr_res.shape[0]}")
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    tried.append((thr, nc, type(e).__name__))
            if success:
                break

        if not success:
            print(f"[KMS][WARN] KMeansSMOTE failed after tries={tried[:6]}... | last_err={type(last_err).__name__}: {last_err}")
            print("[KMS][WARN] Use original train attack windows (no resampling).")
            Xw_tr_res, yt_tr_res = Xw_tr_a, yt_tr_a
    model_type = build_model(cfg, feature_dim, num_classes=K_attack).to(cfg.device)

    if cfg.init_from_stage1:
        sd = safe_torch_load_state_dict(best_bin, map_location="cpu")
        for kk in list(sd.keys()):
            if kk.startswith("fc."):
                sd.pop(kk)
        missing, unexpected = model_type.load_state_dict(sd, strict=False)
        print("[Stage-2] init_from_stage1=True | missing:", len(missing), "unexpected:", len(unexpected))

    w_type = balanced_weights(yt_tr_res, K=K_attack)
    crit_type = nn.CrossEntropyLoss(weight=torch.tensor(w_type, dtype=torch.float32, device=cfg.device))

    dl_tr_type = DataLoader(WindowDataset(Xw_tr_res, yt_tr_res), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_va_type = DataLoader(WindowDataset(Xw_va_a, yt_va_a), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    def eval_attack_multiclass(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        ys, ps = [], []
        for xb, yb0 in loader:
            xb = xb.to(cfg.device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            ys.append(yb0.numpy())
            ps.append(pred.detach().cpu().numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        out = {
            "acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(K_attack)), zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", labels=list(range(K_attack)), zero_division=0)),
        }
        return out

    best_type = "best_ton_exp2_kms_stage2_types.pt"
    train_loop(
        model_type, dl_tr_type, dl_va_type, cfg,
        epochs=cfg.epochs_type,
        criterion=crit_type,
        eval_fn=lambda m, l: eval_attack_multiclass(m, l),
        best_path=best_type,
        best_key="macro_f1",
    )

    # =========================
    # Unknown / 拒识评估（在 test 全量窗口上）
    # =========================
    # 1) Stage-1 attack probability
    prob_attack_te = predict_binary_prob(model_bin, Xw_te, cfg.device, batch_size=cfg.batch_size)
    pred_attack_te = prob_attack_te >= cfg.attack_prob_thr

    # 2) Stage-2 maxprob & pred for windows that were predicted as attack
    idx_attack_pred = np.where(pred_attack_te)[0]
    maxprob_te = np.zeros((len(Xw_te),), dtype=np.float32)
    pred_type_te = np.zeros((len(Xw_te),), dtype=np.int64)

    if len(idx_attack_pred) > 0:
        mp, pr = predict_type_maxprob_and_pred(model_type, Xw_te[idx_attack_pred], cfg.device, batch_size=cfg.batch_size)
        maxprob_te[idx_attack_pred] = mp.astype(np.float32)
        pred_type_te[idx_attack_pred] = pr.astype(np.int64)

    # 3) auto tau from val known attacks
    if cfg.unknown_tau is not None:
        tau = float(cfg.unknown_tau)
        print(f"\n[Unknown tau] manual => tau={tau:.4f}")
    else:
        # 用 val 的 known attack windows（不含 injection）来标定阈值
        idx_val_known = np.where((yb_va == 1) & (yt_va >= 0))[0]
        if len(idx_val_known) == 0:
            tau = 0.5
            print("\n[Unknown tau][WARN] no val known-attack windows, fallback tau=0.5")
        else:
            mp_val, _ = predict_type_maxprob_and_pred(model_type, Xw_va[idx_val_known], cfg.device, batch_size=cfg.batch_size)
            q = max(0.0, min(1.0, 1.0 - float(cfg.known_keep_coverage)))
            tau = float(np.quantile(mp_val, q=q))
            print(f"\n[Unknown tau] auto from val | keep_coverage={cfg.known_keep_coverage:.3f} => tau={tau:.4f}")

    pred_unknown_te = pred_attack_te & (maxprob_te < tau)

    # ground-truth unknown windows: injection attack windows
    injection_mask = (yb_te == 1) & (tlast_te == cfg.unknown_token.lower())

    inj_total = int(injection_mask.sum())
    inj_pred_norm = int(np.sum(injection_mask & (~pred_attack_te)))
    inj_pred_unk = int(np.sum(injection_mask & pred_unknown_te))
    inj_pred_known = int(np.sum(injection_mask & pred_attack_te & (~pred_unknown_te)))

    # 进一步统计：inj_pred_known 被分到哪些 known attack
    inj_known_hist = {}
    if inj_pred_known > 0:
        inj_known_types = pred_type_te[injection_mask & pred_attack_te & (~pred_unknown_te)]
        for k_id, c in zip(*np.unique(inj_known_types, return_counts=True)):
            inj_known_hist[known_attack_types[int(k_id)]] = int(c)

    print("\n========== Unknown(Injection) Routing (Final Pipeline) ==========")
    if inj_total == 0:
        print("[WARN] test 中没有 injection windows（检查 unknown_token/窗口标签模式/stride）")
    else:
        print(f"Injection total windows: {inj_total}")
        print(f"  -> Normal : {inj_pred_norm} ({inj_pred_norm / inj_total:.4f})")
        print(f"  -> Unknown: {inj_pred_unk} ({inj_pred_unk / inj_total:.4f})")
        print(f"  -> Known  : {inj_pred_known} ({inj_pred_known / inj_total:.4f})")
        if inj_known_hist:
            print("  [Injection -> Known breakdown]")
            for name, c in sorted(inj_known_hist.items(), key=lambda x: -x[1]):
                print(f"    {name}: {c}")

    # Unknown metric (Unknown as positive class) on all windows
    y_true_unk = injection_mask.astype(np.int64)
    y_pred_unk = pred_unknown_te.astype(np.int64)
    unk_p = precision_score(y_true_unk, y_pred_unk, zero_division=0)
    unk_r = recall_score(y_true_unk, y_pred_unk, zero_division=0)
    unk_f1 = f1_score(y_true_unk, y_pred_unk, zero_division=0)

    # Known-only Macro-F1 + coverage (only on known attack windows)
    known_mask = (yb_te == 1) & (yt_te >= 0)
    known_total = int(known_mask.sum())
    # covered = predicted as attack and not rejected
    covered_mask = known_mask & pred_attack_te & (~pred_unknown_te)
    coverage = float(covered_mask.sum() / max(1, known_total))

    if covered_mask.sum() > 0:
        y_true_known = yt_te[covered_mask]
        y_pred_known = pred_type_te[covered_mask]
        known_macro_f1 = float(f1_score(y_true_known, y_pred_known, average="macro", labels=list(range(K_attack)), zero_division=0))
    else:
        known_macro_f1 = 0.0

    print("\n========== Metrics Summary ==========")
    print(f"[Attack vs Normal] F1={te_bin['f1']:.4f} P={te_bin['p']:.4f} R={te_bin['r']:.4f} AUROC={te_bin.get('auroc', float('nan')):.4f} thr={cfg.attack_prob_thr}")
    print(f"[Unknown(Injection)] P={unk_p:.4f} R={unk_r:.4f} F1={unk_f1:.4f} | inj->Normal={inj_pred_norm/max(1,inj_total):.4f}")
    print(f"[Known-only] Macro-F1={known_macro_f1:.4f} | coverage={coverage:.4f} | tau={tau:.4f}")

    # 可选：输出 known attack 的详细 report（只在 covered 的 windows 上）
    if covered_mask.sum() > 0:
        try:
            print("\n[Stage-2 ClassificationReport][Known attack windows only | covered]")
            print(classification_report(y_true_known, y_pred_known, labels=list(range(K_attack)), target_names=known_attack_types, zero_division=0, digits=4))
        except Exception:
            pass

    print(f"\n[OK] Saved: {best_bin} | {best_type}")


if __name__ == "__main__":
    main()
