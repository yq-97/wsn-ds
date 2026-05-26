import os
import glob
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, recall_score
)

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model_doublecomparison import AnomalyDetectionModel

# ====================== 基本配置 ======================
config = {
    "feature_dim": 16,        # 会在加载数据后自动更新
    "model_dim": 128,
    "tcn_layers": 2,
    "transformer_layers": 3,
    "nheads": 4,
    "dropout": 0.4,
    "max_len": 64,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 6,
    "gradient_clip": 1.0,
    "num_classes": 5,
    "window": 64,
    "stride": 64,
    "maj_threshold": 0.55,
    "aug_shift_max": 8,
    "aug_prob": 0.90,
}

# ====================== 固定随机种子 ======================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ====================== 读取 parquet 文件夹 ======================
def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    parquet_files = [
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if f.endswith(".parquet") and not f.endswith(".parquet.crc")
    ]
    print(f"📂 在 {folder_path} 中发现 {len(parquet_files)} 个有效 Parquet 文件")
    dfs = []
    for file in parquet_files:
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            dfs.append(df)
            print(f"✅ 已读取 {os.path.basename(file)}，包含 {len(df)} 条记录")
        except Exception as e:
            print(f"⚠️ 跳过 {file}: {e}")
    if not dfs:
        raise RuntimeError(f"{folder_path} 下没有合法 parquet 数据")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"✅ 合并后总记录数: {len(merged)}")
    return merged

# ====================== 窗口数据集 ======================
class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, M: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.M = torch.from_numpy(M).bool()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]

# ====================== A2 + B4 滑窗构造 ======================
def make_windows_A2(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
    feature_cols,
    window: int,
    stride: int,
    maj_thr: float,
    aug_shift_max: int,
    aug_prob: float,
    idx_to_name,
    rare_ids=None,
):
    if rare_ids is None:
        rare_ids = set()
    df = df.sort_values([id_col, time_col])
    groups = df.groupby(id_col)

    X_list, y_list, M_list = [], [], []
    num_classes = len(idx_to_name)

    for sid, g in groups:
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        lab = g[label_col].to_numpy(dtype=np.int64)
        L = len(g)
        if L == 0:
            continue
        s = 0
        while s < L:
            e = min(s + window, L)
            x_real = feat[s:e]
            y_real = lab[s:e]
            valid_len = e - s

            # padding + mask
            if valid_len < window:
                pad_len = window - valid_len
                pad = np.zeros((pad_len, feat.shape[1]), dtype=np.float32)
                x_win = np.concatenate([x_real, pad], axis=0)
                pad_mask = np.ones((window,), dtype=bool)
                pad_mask[:valid_len] = False
            else:
                x_win = x_real
                pad_mask = np.zeros((window,), dtype=bool)

            # A2 窗口标签：稀有类优先 + 多数投票
            if valid_len > 0:
                cnt = np.bincount(y_real.astype(np.int64), minlength=num_classes)
                present_rare = [(r, cnt[r]) for r in rare_ids if cnt[r] > 0]
                if present_rare:
                    # 窗口里只要有稀有类，就选出现次数最多的稀有类
                    y_win = max(present_rare, key=lambda z: z[1])[0]
                else:
                    ratio = cnt / (cnt.sum() + 1e-12)
                    if ratio.max() >= maj_thr:
                        y_win = int(np.argmax(cnt))
                    else:
                        # 不满足阈值也不丢窗口，退回出现最多的类
                        y_win = int(np.argmax(cnt))
            else:
                y_win = 0

            X_list.append(x_win)
            y_list.append(y_win)
            M_list.append(pad_mask)
            s += stride

    X_win = np.stack(X_list, axis=0) if X_list else np.zeros((0, window, len(feature_cols)), np.float32)
    y_win = np.asarray(y_list, dtype=np.int64) if y_list else np.zeros((0,), np.int64)
    M_win = np.stack(M_list, axis=0) if M_list else np.zeros((0, window), np.bool_)

    counts = dict((int(c), int(v)) for c, v in Counter(y_win).items())
    print("[A2] window counts before B4:", counts)

    # B4：对稀有类时间平移增强（只在训练集）
    if y_win.size > 0 and aug_shift_max > 0 and aug_prob > 0:
        median_cnt = int(np.median(list(counts.values()))) if counts else 0
        rare_classes = {c for c, v in counts.items() if c in (rare_ids or set()) and v < median_cnt}
        if rare_classes:
            X_aug, y_aug, M_aug = [], [], []
            rng = np.random.default_rng()
            for x, y, m in zip(X_list, y_list, M_list):
                if y in rare_classes and rng.random() < aug_prob:
                    valid_idx = np.where(m == False)[0]
                    if valid_idx.size == 0:
                        continue
                    s0, e0 = valid_idx[0], valid_idx[-1] + 1
                    k = int(rng.integers(1, aug_shift_max + 1))
                    seg = x[s0:e0].copy()
                    seg = np.roll(seg, shift=k, axis=0)
                    x2 = x.copy()
                    x2[s0:e0] = seg
                    X_aug.append(x2)
                    y_aug.append(y)
                    M_aug.append(m.copy())
            if X_aug:
                print(f"[B4] augmented {len(X_aug)} windows for rare classes {sorted(rare_classes)}")
                X_win = np.concatenate([X_win, np.stack(X_aug, axis=0)], axis=0)
                y_win = np.concatenate([y_win, np.asarray(y_aug, dtype=np.int64)], axis=0)
                M_win = np.concatenate([M_win, np.stack(M_aug, axis=0)], axis=0)

    print("[A2/B4] final windows:", X_win.shape, y_win.shape, M_win.shape)
    if y_win.size:
        counts2 = dict((int(c), int(v)) for c, v in Counter(y_win).items())
        print("[A2/B4] window counts after B4:", counts2)
    return X_win, y_win, M_win

def infer_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for x, y, m in loader:
            x = x.to(device)
            m = m.to(device)
            logits = model(x, pad_mask=m)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)

def infer_logits(model, loader, device):
    """Return raw logits (no softmax) and labels from a dataloader."""
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for x, y, m in loader:
            x = x.to(device)
            m = m.to(device)
            logits = model(x, pad_mask=m)
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_y, axis=0)

def infer_logits_feats(model, loader, device):
    model.eval()
    all_logits, all_feats, all_y = [], [], []
    with torch.no_grad():
        for x, y, m in loader:
            x = x.to(device)
            m = m.to(device)
            logits, feats = model(x, pad_mask=m, return_features=True)
            all_logits.append(logits.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            all_y.append(y.numpy())
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_y, axis=0),
    )

def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score: E(x) = -T * logsumexp(logits/T). Higher => more OOD/unknown."""
    x = logits / T
    m = np.max(x, axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.sum(np.exp(x - m), axis=1))
    return -T * lse

# ====================== 主训练流程 ======================
def main():
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    start_time = datetime.now()
    run_tag = start_time.strftime("%Y%m%d-%H%M%S")
    log_path = save_dir / "experiments_log.txt"

    def append_log(text: str):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    append_log("")
    append_log(f"========== EXPERIMENT {run_tag} ==========")
    append_log(f"Started    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------ 1. 加载原始训练/测试数据 ------------
    print("🚀 开始加载 WSN-DS 数据集...")
    train_df = load_parquet_folder(r"C:\Users\yeqing\PycharmProjects\pythonProject\WSN-DS-main\newtrain.parquet")
    test_df = load_parquet_folder(r"C:\Users\yeqing\PycharmProjects\pythonProject\WSN-DS-main\test.parquet")

    print("训练集攻击类型分布：\n", train_df["Attack_type"].value_counts())
    print("测试集攻击类型分布：\n", test_df["Attack_type"].value_counts())

    ID_COL = "id"
    TIME_COL = "Time"
    LABEL_COL = "Attack_type"

    UNKNOWN_FULL = 2  # Grayhole 在你的数据里就是 2
    ALLOW_KNOWN_REJECT = 0.05  # 已知类允许被拒识的比例（用来定 tau_unknown）
    NORMAL_FPR = 0.05  # 正常流量误报率（用来定 tau_attack）
    SAFE_NORMAL_FPR = 0.01  # Safe-Normal Gate：正常被判为“非安全正常”的比例（越大越激进）

    PULL_NORM_FPR = 0.01  # 只允许 1% 的“真实正常”被我们从 Normal 拽走（控制二分类不崩）
    REJECT_ATK_FPR = 0.02  # 只允许 2% 的“真实已知攻击”被拒识成 Unknown（控制 known-only 不崩）

    SAFE_NORMAL_MD_FPR = 0.03  # 建议 0.02~0.05，越大越“更不让进Normal”
    COV_EPS = 1e-3  # 协方差正则，防止奇异

    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TIME_COL, LABEL_COL]]

    # ------------ 2. 标准化 + 填充缺失 ------------
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    print("🔧 数据预处理...")
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    # ------------ 3. 已知类映射（full -> known），Unknown 不参与训练 ------------
    # train_df 来自 newtrain.parquet（已经没有 2 了）
    known_full = sorted([int(x) for x in train_df[LABEL_COL].unique().tolist()])
    assert UNKNOWN_FULL not in known_full, f"newtrain 里居然还有 Unknown={UNKNOWN_FULL}，先检查数据集！"

    full2known = {full: i for i, full in enumerate(known_full)}
    known2full = {i: full for full, i in full2known.items()}

    K = len(known_full)  # 这里应为 4
    config["num_classes"] = K
    print("[Known full labels]", known_full, "=> K =", K, "| Unknown(full) =", UNKNOWN_FULL)
    print("[Mapping full2known]", full2known)

    # 训练用 known 标签
    train_df[LABEL_COL] = train_df[LABEL_COL].map(full2known).astype(int)

    # test 保留 full 标签（用于后续 unknown 评估），不要覆盖
    test_df["Attack_full"] = test_df[LABEL_COL].astype(int)

    # rare_ids：对训练空间而言，非0 都是攻击
    rare_ids = {c for c in range(K) if c != 0}
    idx_to_name_known = [str(known2full[i]) for i in range(K)]
    print("[Classes(Known)]", K, "=>", idx_to_name_known, "| rare_ids:", rare_ids)

    # ------------ 4. 按 A2+B4 构造训练/验证窗口 ------------
    W = config["window"]
    ST = config["stride"]
    TH = config["maj_threshold"]

    X_all, y_all, M_all = make_windows_A2(
        train_df, ID_COL, TIME_COL, LABEL_COL, feature_cols,
        W, ST, TH,
        config["aug_shift_max"], config["aug_prob"],
        idx_to_name_known, rare_ids=rare_ids,
    )
    X_tr, X_va, y_tr, y_va, M_tr, M_va = train_test_split(
        X_all, y_all, M_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    test_df_full = test_df.copy()
    test_df_full[LABEL_COL] = test_df_full["Attack_full"].astype(int)

    idx_to_name_full = [str(i) for i in range(5)]  # 0..4
    rare_ids_full = {1, 2, 3, 4}

    X_te, y_te_full, M_te = make_windows_A2(
        test_df_full, ID_COL, TIME_COL, LABEL_COL, feature_cols,
        W, ST, TH,
        0, 0.0,
        idx_to_name_full, rare_ids=rare_ids_full,
    )

    print("[Final window counts]", dict(Counter(y_tr.tolist())))
    config["feature_dim"] = X_tr.shape[2]
    print(f"特征维度更新为: {config['feature_dim']}")

    train_dataset = WindowsDataset(X_tr, y_tr, M_tr)
    val_dataset = WindowsDataset(X_va, y_va, M_va)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    gc.collect()

    # ------------ 5. 构建模型 ------------
    model = AnomalyDetectionModel(
        feature_dim=config["feature_dim"],
        model_dim=config["model_dim"],
        num_classes=config["num_classes"],
        tcn_layers=config["tcn_layers"],
        transformer_layers=config["transformer_layers"],
        nheads=config["nheads"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(config["device"])

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------ 6. 优化器 + 调度器 + 损失函数 ------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["learning_rate"] * 0.1,
    )

    # ——多分类 CrossEntropy + 类别权重（就是你 best 那版的核心）——
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(K),
        y=y_tr,
    ).astype(np.float32)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=config["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print("[class_weights]", {i: float(w) for i, w in enumerate(class_weights)})

    # ------------ 7. 训练循环 ------------
    best_f1 = 0.0
    best_epoch = -1
    no_improve = 0
    all_train_losses, all_val_losses = [], []
    all_train_f1s, all_val_f1s = [], []

    labels_all = np.arange(K)

    for epoch in range(config["epochs"]):
        # === Train ===
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        from tqdm.auto import tqdm
        for inputs, labels, pad_mask in tqdm(
            train_loader,
            desc=f"[Train] {epoch+1}/{config['epochs']}",
            leave=False,
            dynamic_ncols=True,
        ):
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])
            pad_mask = pad_mask.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs, pad_mask=pad_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())

        train_avg_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_targets, train_preds, average="macro", labels=labels_all, zero_division=0)
        print(f"Epoch {epoch+1} - 训练 - 损失: {train_avg_loss:.4f}, F1: {train_f1:.4f}")
        print("训练集混淆矩阵:\n", confusion_matrix(train_targets, train_preds))
        all_train_losses.append(train_avg_loss)
        all_train_f1s.append(train_f1)

        # === Val ===
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels, pad_mask in val_loader:
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])
                pad_mask = pad_mask.to(config["device"])

                outputs = model(inputs, pad_mask=pad_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_targets.extend(labels.detach().cpu().numpy().tolist())

        val_avg_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average="macro", labels=labels_all, zero_division=0)
        print(f"Epoch {epoch+1}/{config['epochs']} - 验证 - 损失: {val_avg_loss:.4f}, F1: {val_f1:.4f}")
        print("验证集混淆矩阵:\n", confusion_matrix(val_targets, val_preds))

        labels_all = list(range(K))

        prec, rec, f1c, sup = precision_recall_fscore_support(
            val_targets,
            val_preds,
            labels=labels_all,
            average=None,
            zero_division=0
        )

        print("\n[Per-Class on VAL]")
        for i in range(K):
            name = idx_to_name_known[i]
            print(f"{name:<5} P={prec[i]:.3f} R={rec[i]:.3f} F1={f1c[i]:.3f} N={int(sup[i])}")

        scheduler.step()
        all_val_losses.append(val_avg_loss)
        all_val_f1s.append(val_f1)

        # 早停 + 保存最优
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": val_f1,
                    "config": config,
                },
                save_dir / "best_model.pth",
            )
            print(f"💾 保存最佳模型，F1: {best_f1:.4f}")
        else:
            no_improve += 1
            print(f"⏸ 模型性能未提升 ({no_improve}/{config['patience']})")
            if no_improve >= config["patience"]:
                print(f"🛑 提前停止：{config['patience']} 个 epoch 未改善")
                break

    # 保存训练曲线
    np.savez(
        save_dir / "training_history.npz",
        train_losses=np.array(all_train_losses),
        val_losses=np.array(all_val_losses),
        train_f1s=np.array(all_train_f1s),
        val_f1s=np.array(all_val_f1s),
    )

    # ------------ 8. Unknown/拒识机制：先在 VAL 上定阈值，再在 TEST 上评估（加入 Safe-Normal Gate） ------------

    # 重新加载 best_model

    ckpt = torch.load(save_dir / "best_model.pth", map_location=config["device"])

    model.load_state_dict(ckpt["model_state_dict"])

    normal_known_id = int(full2known.get(0, 0))

    # =======================
    # 1) VAL：标定阈值（已知类）
    # =======================

    val_logits, val_feats, val_y = infer_logits_feats(model, val_loader, config["device"])
    val_probs = softmax_np(val_logits)

    E_val = energy_score(val_logits, T=1.0)

    pnormal_val = val_probs[:, normal_known_id]

    # ---------- Feature-prototype for Safe-Normal Gate ----------
    val_pred = val_probs.argmax(axis=1)
    mask_good_norm = (val_y == normal_known_id) & (val_pred == normal_known_id)  # 只用“预测也为正常”的干净正常来定阈值

    # class prototypes in feature space
    proto_feat = []
    for k in range(K):
        mk = (val_y == k)
        if mk.sum() == 0:
            proto_feat.append(np.zeros((val_feats.shape[1],), dtype=np.float32))
        else:
            proto_feat.append(val_feats[mk].mean(axis=0))
    proto_feat = np.stack(proto_feat, axis=0)  # [K, D]
    proto_feat_norm = proto_feat / (np.linalg.norm(proto_feat, axis=1, keepdims=True) + 1e-12)

    val_feats_norm = val_feats / (np.linalg.norm(val_feats, axis=1, keepdims=True) + 1e-12)
    sim_feat_val = val_feats_norm @ proto_feat_norm.T
    sim0_feat_val = sim_feat_val[:, normal_known_id]
    sim_other_feat_val = np.max(np.delete(sim_feat_val, normal_known_id, axis=1), axis=1)
    margin_feat_val = sim0_feat_val - sim_other_feat_val

    # ========= (NEW) Mahalanobis Normal-Gate in feature space =========
    def _fit_gaussian_inv(feats: np.ndarray, eps: float = COV_EPS):
        mu = feats.mean(axis=0)
        cov = np.cov(feats, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * eps
        inv = np.linalg.inv(cov)
        return mu, inv

    def _mahalanobis(feats: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray):
        diff = feats - mu
        return np.einsum("ni,ij,nj->n", diff, inv_cov, diff)

    # 用“干净正常”(val 中被模型判对 Normal 的样本) 拟合 Normal 分布
    feats_norm_clean = val_feats[mask_good_norm]
    mu_norm, inv_cov_norm = _fit_gaussian_inv(feats_norm_clean, eps=COV_EPS)
    md_val_clean = _mahalanobis(feats_norm_clean, mu_norm, inv_cov_norm)
    tau_md_pull = float(np.quantile(md_val_clean, 1.0 - SAFE_NORMAL_MD_FPR))  # md 越大越不像正常

    # Safe-Normal thresholds: 允许 SAFE_NORMAL_FPR 比例的“正常”被拒
    tau_sim0_feat_safe = float(np.quantile(sim0_feat_val[mask_good_norm], SAFE_NORMAL_FPR))
    tau_margin_feat_safe = float(np.quantile(margin_feat_val[mask_good_norm], SAFE_NORMAL_FPR))

    # 同时把你原来的 p0/E 阈值方向改成“拒掉尾部”，别用 99% 那种写法
    tau_p0_safe = float(np.quantile(pnormal_val[mask_good_norm], SAFE_NORMAL_FPR))  # p0 的低尾
    tau_E_safe = float(np.quantile(E_val[mask_good_norm], 1.0 - SAFE_NORMAL_FPR))  # E 的高尾（E越大越不像正常）

    # Unknown(Energy) 阈值：允许 ALLOW_KNOWN_REJECT 的已知样本被拒识

    tau_energy = float(np.quantile(E_val, 1.0 - ALLOW_KNOWN_REJECT))

    # Attack 阈值：允许 NORMAL_FPR 的正常样本被误报为攻击

    s_attack_val = 1.0 - pnormal_val

    mask_norm_val = (val_y == normal_known_id)

    tau_attack = float(np.quantile(s_attack_val[mask_norm_val], 1.0 - NORMAL_FPR))

    # ---- (NEW) Logits-prototype features for safer Normal gate ----
    # Build class prototypes (mean logits) on VAL (known classes only).
    proto_logits = []
    for k in range(K):
        mk = (val_y == k)
        if mk.sum() == 0:
            proto_logits.append(np.zeros((val_logits.shape[1],), dtype=np.float32))
        else:
            proto_logits.append(val_logits[mk].mean(axis=0))
    proto_logits = np.stack(proto_logits, axis=0)  # [K, K]
    proto_norm = proto_logits / (np.linalg.norm(proto_logits, axis=1, keepdims=True) + 1e-12)

    val_logits_norm = val_logits / (np.linalg.norm(val_logits, axis=1, keepdims=True) + 1e-12)
    sim_val = val_logits_norm @ proto_norm.T  # [N, K] cosine(logits, proto)
    sim0_val = sim_val[:, normal_known_id]
    sim_other_max_val = np.max(np.delete(sim_val, normal_known_id, axis=1), axis=1)
    margin0_val = sim0_val - sim_other_max_val

    # Safe-Normal extra thresholds from VAL-Normal (lower tail):
    tau_sim0_safe = float(np.quantile(sim0_val[mask_norm_val], SAFE_NORMAL_FPR))
    tau_margin0_safe = float(np.quantile(margin0_val[mask_norm_val], SAFE_NORMAL_FPR))

    # ====== (NEW) 仅用于“叠加 Unknown”的阈值：保证不伤你原本最优分类 ======

    # 1) Normal 拉出阈值：只拉出 val-normal 中最可疑的 PULL_NORM_FPR 部分
    p2_val = np.partition(val_probs, -2, axis=1)[:, -2]  # 第二大概率
    margin0_val_soft = pnormal_val - p2_val  # Normal 置信度间隔（越小越不可信）
    tau_margin0_pull = float(np.quantile(margin0_val_soft[mask_norm_val], PULL_NORM_FPR))

    tau_E_norm_pull = float(np.quantile(E_val[mask_norm_val], 1.0 - PULL_NORM_FPR))  # 正常能量高尾：可疑
    tau_attack_pull = float(np.quantile(s_attack_val[mask_norm_val], 1.0 - PULL_NORM_FPR))  # (1-p0)高尾：可疑

    # 2) Attack 拒识阈值：只拒识 val-attack 中最不可信的 REJECT_ATK_FPR 部分
    mask_atk_val = ~mask_norm_val
    probs_non0_val = val_probs.copy()
    probs_non0_val[:, normal_known_id] = -1.0
    atk_top1_val = probs_non0_val.max(axis=1)
    atk_top2_val = np.partition(probs_non0_val, -2, axis=1)[:, -2]
    atk_gap_val = atk_top1_val - atk_top2_val

    tau_atk_conf_rej = float(np.quantile(atk_top1_val[mask_atk_val], REJECT_ATK_FPR))  # 攻击 top1 低尾
    tau_atk_gap_rej = float(np.quantile(atk_gap_val[mask_atk_val], REJECT_ATK_FPR))  # 攻击 gap 低尾
    tau_E_atk_rej = float(np.quantile(E_val[mask_atk_val], 1.0 - REJECT_ATK_FPR))  # 攻击能量高尾

    print(
        f"[Overlay THR] "
        f"tau_attack_pull={tau_attack_pull:.4f} | "
        f"tau_margin0_pull={tau_margin0_pull:.4f} | "
        f"tau_E_norm_pull={tau_E_norm_pull:.4f} | "
        f"tau_atk_conf_rej={tau_atk_conf_rej:.4f} | "
        f"tau_atk_gap_rej={tau_atk_gap_rej:.4f} | "
        f"tau_E_atk_rej={tau_E_atk_rej:.4f}"
    )

    print(
        f"[Thresholds] "
        f"tau_energy(E>{tau_energy:.4f}) | "
        f"tau_attack(1-p0>{tau_attack:.4f}) | "
        f"tau_p0_safe(p0>={tau_p0_safe:.4f}) | "
        f"tau_E_safe(E<={tau_E_safe:.4f})"
    )
    # =======================
    # 2) TEST：推理 + 评估
    # =======================

    test_dataset = WindowsDataset(X_te, y_te_full, M_te)  # y_te_full 是 full 标签（含 Unknown）
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    te_logits, te_feats, te_y_full = infer_logits_feats(model, test_loader, config["device"])
    te_probs_known = softmax_np(te_logits)

    E_te = energy_score(te_logits, T=1.0)

    pred_known = te_probs_known.argmax(axis=1)

    pnormal_te = te_probs_known[:, normal_known_id]

    s_attack_te = 1.0 - pnormal_te

    # =======================
    # 2) TEST：推理 + 叠加 Unknown（不破坏你原本最优分类）
    # =======================

    # 先做“原本最优”的 baseline 多分类（绝不 force non0）
    pred_known = te_probs_known.argmax(axis=1)
    pred_full = np.array([known2full[int(k)] for k in pred_known], dtype=int)

    # 计算基础分数
    pnormal_te = te_probs_known[:, normal_known_id]
    s_attack_te = 1.0 - pnormal_te

    te_feats_norm = te_feats / (np.linalg.norm(te_feats, axis=1, keepdims=True) + 1e-12)
    md_te = _mahalanobis(te_feats, mu_norm, inv_cov_norm)

    sim_feat_te = te_feats_norm @ proto_feat_norm.T
    sim0_feat_te = sim_feat_te[:, normal_known_id]
    sim_other_feat_te = np.max(np.delete(sim_feat_te, normal_known_id, axis=1), axis=1)
    margin_feat_te = sim0_feat_te - sim_other_feat_te

    # Normal 置信度间隔（softmax margin）
    p2_te = np.partition(te_probs_known, -2, axis=1)[:, -2]
    margin0_te_soft = pnormal_te - p2_te

    # logits-prototype 的 sim/margin（你原来算过，就保留这段）
    te_logits_norm = te_logits / (np.linalg.norm(te_logits, axis=1, keepdims=True) + 1e-12)
    sim_te = te_logits_norm @ proto_norm.T
    sim0_te = sim_te[:, normal_known_id]
    sim_other_max_te = np.max(np.delete(sim_te, normal_known_id, axis=1), axis=1)
    margin0_te_proto = sim0_te - sim_other_max_te

    # Attack 侧 top1/top2 gap（softmax）
    probs_non0_te = te_probs_known.copy()
    probs_non0_te[:, normal_known_id] = -1.0
    atk_top1_te = probs_non0_te.max(axis=1)
    atk_top2_te = np.partition(probs_non0_te, -2, axis=1)[:, -2]
    atk_gap_te = atk_top1_te - atk_top2_te

    # ---------- 叠加 Unknown：两类触发 ----------
    # A) 最高优先级：把“预测为 Normal 的可疑样本”直接改成 Unknown（优先降低 Grayhole->Normal）
    mask_pred_norm = (pred_full == 0)
    mask_pull_from_normal = mask_pred_norm & (
            (s_attack_te > tau_attack_pull) |
            (margin0_te_soft < tau_margin0_pull) |
            (E_te > tau_E_norm_pull) |
            (md_te > tau_md_pull) |  # (NEW) Mahalanobis：高置信度但特征分布异常的“伪正常”
            (margin0_te_proto < tau_margin0_safe) |
            (sim0_te < tau_sim0_safe) |
            # ===== 新增：用“特征空间”的原型相似度来拉 Unknown（专门打 Grayhole->Normal）=====
            (sim0_feat_te < tau_sim0_feat_safe) |
            (margin_feat_te < tau_margin_feat_safe)
    )

    # B) 次优先级：把“预测为攻击但很不可信”的样本改成 Unknown（降低 Grayhole->Other Attack，提升 Unknown）
    mask_pred_atk = (pred_full != 0)
    mask_reject_attack = mask_pred_atk & (
            (atk_top1_te < tau_atk_conf_rej) |
            (atk_gap_te < tau_atk_gap_rej) |
            (E_te > tau_E_atk_rej)
    )

    pred_is_unknown_final = mask_pull_from_normal | mask_reject_attack
    pred_full[pred_is_unknown_final] = UNKNOWN_FULL

    # =======================
    # 3) 指标与统计
    # =======================
    gt_is_unknown = (te_y_full == UNKNOWN_FULL)
    gt_is_attack = (te_y_full != 0)

    pred_is_attack_final = (pred_full != 0)

    p_attack = precision_score(gt_is_attack.astype(int), pred_is_attack_final.astype(int), zero_division=0)

    r_attack = recall_score(gt_is_attack.astype(int), pred_is_attack_final.astype(int), zero_division=0)

    f1_attack = f1_score(gt_is_attack.astype(int), pred_is_attack_final.astype(int), zero_division=0)

    auc_attack = roc_auc_score(gt_is_attack.astype(int), s_attack_te)

    print(f"[Attack vs Normal] P={p_attack:.4f} R={r_attack:.4f} F1={f1_attack:.4f} AUROC={auc_attack:.4f}")

    # --- Attack vs Normal 2x2 confusion matrix ---
    gt_ab = gt_is_attack.astype(int)  # 0=Normal, 1=Attack
    pd_ab = pred_is_attack_final.astype(int)

    cm = confusion_matrix(gt_ab, pd_ab, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"[Attack vs Normal CM] [[TN FP],[FN TP]] = {cm.tolist()}")
    print(f"[Attack vs Normal ERR] Normal->Attack(FP)={fp} | Attack->Normal(FN)={fn}")
    print(f"[Attack vs Normal RATE] FPR={fp / (fp + tn + 1e-12):.4f} | FNR={fn / (fn + tp + 1e-12):.4f}")

    p_u = precision_score(gt_is_unknown.astype(int), pred_is_unknown_final.astype(int), zero_division=0)

    r_u = recall_score(gt_is_unknown.astype(int), pred_is_unknown_final.astype(int), zero_division=0)

    f1_u = f1_score(gt_is_unknown.astype(int), pred_is_unknown_final.astype(int), zero_division=0)

    auc_u_energy = roc_auc_score(gt_is_unknown.astype(int), E_te)

    print(f"[Unknown] P={p_u:.4f} R={r_u:.4f} F1={f1_u:.4f} AUROC(E)={auc_u_energy:.4f}")

    # Known-only 宏F1：真实已知 & 未拒识
    mask_known_eval = (~gt_is_unknown) & (~pred_is_unknown_final)

    if mask_known_eval.sum() > 0:
        yk = np.array([full2known.get(int(y), -1) for y in te_y_full[mask_known_eval]], dtype=int)
        pk = np.array([full2known.get(int(y), -1) for y in pred_full[mask_known_eval]], dtype=int)
        macro_f1_known = f1_score(yk, pk, average="macro", labels=list(range(K)), zero_division=0)
        coverage = float(mask_known_eval.mean())
        print(f"[Known-only] Macro-F1={macro_f1_known:.4f} | coverage={coverage:.4f}")

    # Grayhole 去向：Unknown / Normal / Other Attack
    mask_gh = (te_y_full == UNKNOWN_FULL)

    n_gh = int(mask_gh.sum())

    if n_gh > 0:
        gh_as_unknown = int((mask_gh & pred_is_unknown_final).sum())
        gh_as_normal  = int((mask_gh & (pred_full == 0)).sum())
        gh_as_other   = int((mask_gh & (pred_full != 0) & (pred_full != UNKNOWN_FULL)).sum())
        print("")
        print("[Grayhole Breakdown]")
        print(f"Total Grayhole windows      : {n_gh}")
        print(f"Pred as Unknown (reject)    : {gh_as_unknown}  ({gh_as_unknown/max(n_gh,1):.4f})")
        print(f"Pred as Normal (0)          : {gh_as_normal}   ({gh_as_normal/max(n_gh,1):.4f})")
        print(f"Pred as Other Attack (1/3/4): {gh_as_other} ({gh_as_other/max(n_gh,1):.4f})")
        vals, cnts = np.unique(pred_full[mask_gh], return_counts=True)
        print("[Grayhole predicted full label counts]", dict(zip(vals.tolist(), cnts.tolist())))

    print(f"🎯 训练完成！最佳验证 F1: {best_f1:.4f}, epoch={best_epoch}")
if __name__ == "__main__":
    main()
