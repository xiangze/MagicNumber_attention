"""
Lyapunov Spectrum Analyzer for Attention + FNN Models
======================================================
画像データセット (MNIST / FashionMNIST / CIFAR-10 / CIFAR-100) と
独自 CSV・合成データに対応。
学習後に各層間の中間状態遷移のリャプノフスペクトルを計算する。

依存パッケージ:
    pip install torch torchvision matplotlib numpy

使い方:
    # MNIST で学習 → リャプノフ解析
    python lyapunov_model.py --task mnist --epochs 10

    # Fashion-MNIST
    python lyapunov_model.py --task fashion_mnist --N 28 --M 28

    # CIFAR-10
    python lyapunov_model.py --task cifar10 --N 32 --M 96 --n_blocks 6

    # CIFAR-100
    python lyapunov_model.py --task cifar100 --N 32 --M 96 --n_blocks 8

    # 合成 sin データ
    python lyapunov_model.py --task sin --N 16 --M 7

    # CSV (回帰)
    python lyapunov_model.py --task csv --data_path data.csv --N 16 --M 7
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.functional as AF
from torch.utils.data import DataLoader, Subset, TensorDataset

import torchvision
import torchvision.transforms as transforms
HAS_TORCHVISION = True
import itertools
# ============================================================
# 1. モデル定義
# ============================================================
class SelfAttentionLayer(nn.Module):
    """
    単純化した自己注意層。
    softmax(τ · X Xᵀ) X + 残差接続。τ は学習可能。
    """
    def __init__(self, rate: float = 1.0):
        super().__init__()
        self.rate    = rate
        self.log_tau = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau    = torch.exp(self.log_tau)
        scores = (x @ x.t()) * tau
        attn   = torch.softmax(scores - scores.max(), dim=-1)
        return self.rate * (attn @ x) + (1.0 - self.rate) * x

class FNNLayer(nn.Module):
    """tanh(β W x + b)。オプションで残差接続。"""
    def __init__(self, n: int, residual: bool = False, beta: float = 1.0):
        super().__init__()
        self.W        = nn.Parameter(torch.randn(n, n) * 0.1)
        self.b        = nn.Parameter(torch.zeros(n))
        self.residual = residual
        self.beta     = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M) — W は N×N で各トークン列に独立に適用
        h = torch.tanh(self.beta * (self.W @ x) + self.b.unsqueeze(-1))
        return h + x if self.residual else h

class AttentionFNNModel(nn.Module):
    """
    画像 / シーケンス両対応の Attention+FNN モデル。

    画像タスク:
        (C*H*W,) → 線形埋め込み → (N, M) → ブロック列 → 分類ヘッド

    回帰タスク:
        (N, M) → ブロック列 → 回帰ヘッド

    Parameters
    ----------
    n_input        : トークン次元 N
    n_seq          : トークン数 M
    n_classes      : クラス数 (None → 回帰モード)
    img_flat_dim   : 入力画像のフラット次元 (画像タスク時)
    n_blocks       : ブロック数
    attn_per_block : 1 ブロックあたりの Attention 層数
    fnn_per_block  : 1 ブロックあたりの FNN 層数
    residual       : FNN 残差接続フラグ
    beta           : FNN 活性化スケール β
    dropout        : ヘッド直前の Dropout 率
    """
    def __init__(
        self,
        n_input:        int,
        n_seq:          int,
        n_classes:      int   = None,
        img_flat_dim:   int   = None,
        n_blocks:       int   = 5,
        attn_per_block: int   = 3,
        fnn_per_block:  int   = 1,
        residual:       bool  = False,
        beta:           float = 2.0,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.n_input       = n_input
        self.n_seq         = n_seq
        self.n_classes     = n_classes
        self.is_classifier = (n_classes is not None)

        # 画像埋め込み層
        if img_flat_dim is not None:
            self.embed = nn.Sequential(
                nn.Linear(img_flat_dim, n_input * n_seq),
                nn.LayerNorm(n_input * n_seq),
            )
        else:
            self.embed = None

        # Attention + FNN ブロック列
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "fnn_layers": nn.ModuleList([
                    FNNLayer(n_input, residual=residual, beta=beta)
                    for _ in range(fnn_per_block)
                ]),
                "attn_layers": nn.ModuleList([
                    SelfAttentionLayer()
                    for _ in range(attn_per_block)
                ]),
            }))

        # 出力ヘッド
        flat_dim    = n_input * n_seq
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(flat_dim, n_classes if self.is_classifier else n_seq)

    def _apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """単一サンプル (N, M) に全ブロックを適用。"""
        for block in self.blocks:
            for layer in block["fnn_layers"]:
                x = layer(x)
            for layer in block["attn_layers"]:
                x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, img_flat) — 画像モード
            (batch, N, M)     — シーケンスモード
            (N, M)            — 単一サンプル (リャプノフ計算用)
        """
    # 画像モード: 1D が単一、2D がバッチ
        if self.embed is not None:
            single = (x.dim() == 1)
        # シーケンスモード: 2D が単一、3D がバッチ
        else:
            single = (x.dim() == 2)

        if single:
            x = x.unsqueeze(0)

        if self.embed is not None:
            x = self.embed(x)
            x=x.reshape(x.size(0), self.n_input, self.n_seq)

        x   = torch.stack([self._apply_blocks(xi) for xi in x])
        out = self.head(self.drop(x.reshape(x.size(0), -1)))

        if single:
            out = out.squeeze(0)
        return out

# ============================================================
# 2. データセット準備
# ============================================================
IMAGE_DATASETS = {
    "mnist":         ("MNIST",        1, 28, 28,  10),
    "fashion_mnist": ("FashionMNIST", 1, 28, 28,  10),
    "kmnist":        ("KMNIST",       1, 28, 28,  10),
    "emnist":        ("EMNIST",       1, 28, 28,  47),  # balanced split
    "svhn":          ("SVHN",         3, 32, 32,  10),
    "cifar10":       ("CIFAR10",      3, 32, 32,  10),
    "cifar100":      ("CIFAR100",     3, 32, 32, 100),
    "stl10":         ("STL10",        3, 96, 96,  10),
    "eurosat":       ("EuroSAT",      3, 64, 64,  10),
}

LABEL_NAMES = {
    "mnist":         list(map(str, range(10))),
    "fashion_mnist": ["T-shirt","Trouser","Pullover","Dress","Coat",
                      "Sandal","Shirt","Sneaker","Bag","Ankle boot"],
    "cifar10":       ["airplane","automobile","bird","cat","deer",
                      "dog","frog","horse","ship","truck"],
    "cifar100":      [f"c{i}" for i in range(100)],
}

# タスク別のデフォルト N, M
TASK_DEFAULTS = {
    "mnist":         (28, 28),   # 784 px → 28 行×28 列
    "fashion_mnist": (28, 28),
    "cifar10":       (32, 96),   # 3072 px → 32×96
    "cifar100":      (32, 96),
    "sin":           (16,  7),
    "csv":           (16,  7),
}

def load_image_dataset(name, data_root="./data", max_train=None, max_test=None):
    """torchvision からイメージデータセットを読み込む。"""

    ds_name, C, H, W, n_classes = IMAGE_DATASETS[name]
    img_flat = C * H * W

    if C == 1:
        mean, std = (0.5,), (0.5,)
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

    # 学習用: CIFAR は RandomHorizontalFlip を追加
    aug = [transforms.RandomHorizontalFlip()] if C == 3 else []
    tf_train = transforms.Compose(aug + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: x.reshape(-1)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(lambda x: x.reshape(-1)),
    ])

    ds_cls   = getattr(torchvision.datasets, ds_name)
    train_ds = ds_cls(root=data_root, train=True,  download=True, transform=tf_train)
    test_ds  = ds_cls(root=data_root, train=False, download=True, transform=tf_test)

    if max_train is not None:
        train_ds = Subset(train_ds, list(range(min(max_train, len(train_ds)))))
    if max_test is not None:
        test_ds  = Subset(test_ds,  list(range(min(max_test,  len(test_ds)))))

    return train_ds, test_ds, n_classes, img_flat

def make_sin_dataset(n_samples=500, seq_len=7, n_features=16,
                     noise=0.05, device=torch.device("cpu")):
    t = torch.linspace(0, 4 * math.pi, n_samples)
    X = torch.stack([torch.sin(t + i * math.pi / n_features) + noise * torch.randn(n_samples) for i in range(n_features)], dim=1)
    X = X.unsqueeze(-1).expand(-1, -1, seq_len)
    y = torch.sin(t + math.pi / 4)
    X = (X - X.mean()) / (X.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return TensorDataset(X.to(device), y.unsqueeze(-1).expand(-1, seq_len).to(device))

def make_csv_dataset(path, n_features, seq_len, target_col=-1, device=torch.device("cpu")):
    data = []
    with open(path) as f:
        for row in csv.reader(f):
            try:
                data.append([float(v) for v in row])
            except ValueError:
                pass
    arr  = torch.tensor(data, dtype=torch.float32)
    tc   = target_col if target_col >= 0 else arr.shape[1] - 1
    feat = arr[:, [i for i in range(arr.shape[1]) if i != tc]]
    tgt  = arr[:, tc]
    if feat.shape[1] < n_features:
        feat = torch.cat([feat, torch.zeros(feat.shape[0], n_features - feat.shape[1])], 1)
    else:
        feat = feat[:, :n_features]
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    tgt  = (tgt  - tgt.mean())  / (tgt.std()  + 1e-8)
    X    = feat.unsqueeze(-1).expand(-1, -1, seq_len)
    y    = tgt.unsqueeze(-1).expand(-1, seq_len)
    return TensorDataset(X.to(device), y.to(device))

# ============================================================
# 3. トレーニング
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, is_cls):
    model.train()
    total_loss = correct = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        if is_cls:
            correct += (pred.argmax(1) == y_b).sum().item()
        total += X_b.size(0)
    return total_loss / total, (correct / total if is_cls else None)

@torch.no_grad()
def evaluate(model, loader, criterion, device, is_cls):
    model.eval()
    total_loss = correct = total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        total_loss += criterion(pred, y_b).item() * X_b.size(0)
        if is_cls:
            correct += (pred.argmax(1) == y_b).sum().item()
        total += X_b.size(0)
    return total_loss / total, (correct / total if is_cls else None)

def train(model, train_loader, test_loader,
          epochs=20, lr=1e-3, device=torch.device("cpu"), verbose=True):
    is_cls    = model.is_classifier
    criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(epochs):
        tr_l, tr_a = train_epoch(model, train_loader, optimizer, criterion, device, is_cls)
        te_l, te_a = evaluate(model, test_loader, criterion, device, is_cls)
        scheduler.step()
        history["train_loss"].append(tr_l)
        history["test_loss"].append(te_l)
        history["train_acc"].append(tr_a)
        history["test_acc"].append(te_a)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            if is_cls:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"train loss={tr_l:.4f} acc={tr_a*100:.1f}% | "
                      f"test loss={te_l:.4f} acc={te_a*100:.1f}%")
            else:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"train loss={tr_l:.6f} | test loss={te_l:.6f}")
    return history

# ============================================================
# 4. リャプノフスペクトル計算
# ============================================================
def jacobian_autograd(f, x: torch.Tensor) -> torch.Tensor:
    """(N, M) → (NM, NM) のヤコビアン行列を autograd で計算。"""
    N, M = x.shape
    NM   = N * M
    def f_flat(xf):
        return f(xf.reshape(N, M)).reshape(NM)
    return AF.jacobian(f_flat, x.reshape(NM).detach().requires_grad_(True)).detach()

def calc_lyapunov_spectrum(model, x0, tiny=1e-300):
    """
    単一初期状態 x0: (N, M) のリャプノフスペクトルを計算。
    embed 済みの (N, M) 表現を渡すこと。
    """
    model.eval()
    N, M = x0.shape
    NM   = N * M
    x    = x0.clone().detach().float()

    Q        = torch.eye(NM, dtype=x.dtype)
    lyap_sum = torch.zeros(NM, dtype=x.dtype)
    per_layer_results = []
    n_layers = 0

    for b_idx, block in enumerate(model.blocks):
        for l_idx, layer in enumerate(block["fnn_layers"]):
            name = f"B{b_idx}/FNN{l_idx}"
            def f_l(xx, _l=layer): return _l(xx)
            J    = jacobian_autograd(f_l, x)
            x    = layer(x)
            Q, R = torch.linalg.qr(J @ Q, mode="reduced")
            diag = torch.log(torch.clamp(torch.abs(torch.diag(R)), min=tiny))
            lyap_sum += diag
            n_layers += 1
            per_layer_results.append((name, diag.numpy().copy()))

        for a_idx, layer in enumerate(block["attn_layers"]):
            name = f"B{b_idx}/Attn{a_idx}"
            def f_l(xx, _l=layer): return _l(xx)
            J    = jacobian_autograd(f_l, x)
            x    = layer(x)
            Q, R = torch.linalg.qr(J @ Q, mode="reduced")
            diag = torch.log(torch.clamp(torch.abs(torch.diag(R)), min=tiny))
            lyap_sum += diag
            n_layers += 1
            per_layer_results.append((name, diag.numpy().copy()))

    return {
        "global":             (lyap_sum / max(n_layers, 1)).numpy(),
        "per_layer":          per_layer_results,
        "max_lyap_per_layer": [s.max().item() for _, s in per_layer_results],
    }


def embed_samples(model, X: torch.Tensor) -> torch.Tensor:
    """
    画像モデルなら embed 層を通して (n, N, M) に変換する。
    シーケンスモデルはそのまま返す。
    """
    model.eval()
    with torch.no_grad():
        if model.embed is not None:
            emb = model.embed(X.float())
            return emb.reshape(-1, model.n_input, model.n_seq)
        return X

def calc_lyapunov_averaged(model, X_embedded, n_samples=5, verbose=True):
    """
    X_embedded: (n, N, M) — 複数サンプルの平均リャプノフスペクトルを返す。
    """
    idx = torch.randperm(X_embedded.shape[0])[:n_samples]
    all_global, all_max = [], []
    last_res = None
    for i, j in enumerate(idx):
        if verbose:
            print(f"  サンプル {i+1}/{n_samples} 計算中...")
        last_res = calc_lyapunov_spectrum(model, X_embedded[j])
        all_global.append(last_res["global"])
        all_max.append(last_res["max_lyap_per_layer"])

    return {
        "mean_global":        np.mean(all_global, axis=0),
        "std_global":         np.std(all_global,  axis=0),
        "mean_per_layer_max": np.mean(all_max,    axis=0),
        "layer_names":        [n for n, _ in last_res["per_layer"]],
    }

# ============================================================
# 5. 可視化
# ============================================================
@torch.no_grad()
def compute_confusion(model, loader, n_classes, device):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for X, y in loader:
        pred = model(X.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    return cm

def plot_results(history, lyap_result, task, save_prefix="lyap_result",
                 conf_mat=None, label_names=None):
    is_cls   = history["train_acc"][0] is not None
    has_conf = is_cls and conf_mat is not None
    n_rows   = 2 if has_conf else 1

    fig = plt.figure(figsize=(16, 5 * n_rows))

    # --- 上段 3 列 ---
    # (0,0) 損失 & 精度
    ax = fig.add_subplot(n_rows, 3, 1)
    ax.plot(history["train_loss"], label="train loss", color="steelblue")
    ax.plot(history["test_loss"],  label="test loss",  color="coral")
    ax.set_title("Loss curve"); ax.set_xlabel("Epoch"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)
    if is_cls:
        ax2 = ax.twinx()
        ax2.plot(history["train_acc"], "--", color="steelblue", alpha=0.5, label="train acc")
        ax2.plot(history["test_acc"],  "--", color="coral",     alpha=0.5, label="test acc")
        ax2.set_ylim(0, 1); ax2.set_ylabel("Accuracy"); ax2.legend(loc="center right")

    # (0,1) グローバルリャプノフスペクトル
    ax = fig.add_subplot(n_rows, 3, 2)
    mg  = lyap_result["mean_global"]
    sg  = lyap_result["std_global"]
    ax.bar(np.arange(len(mg)), mg, yerr=sg, color="mediumslateblue",
           alpha=0.8, capsize=2, error_kw={"linewidth": 0.8})
    ax.axhline(0, color="k", lw=0.8, ls="--")

    # Kaplan-Yorke 次元
    sl = np.sort(mg)[::-1]
    cm_sum = np.cumsum(sl)
    ki = np.searchsorted(-cm_sum, 0)
    ky_str = ""
    if 0 < ki < len(sl) and sl[ki] != 0:
        d_ky   = ki + cm_sum[ki - 1] / abs(sl[ki])
        ky_str = f"  KY≈{d_ky:.1f}"
    ax.set_title(f"Global Lyapunov spectrum ({task}){ky_str}")
    ax.set_xlabel("Index"); ax.set_ylabel("Exponent"); ax.grid(True, alpha=0.3)

    # (0,2) 各層の最大リャプノフ指数
    ax = fig.add_subplot(n_rows, 3, 3)
    names    = lyap_result["layer_names"]
    max_lyap = lyap_result["mean_per_layer_max"]
    colors   = ["steelblue" if "FNN" in n else "coral" for n in names]
    ax.bar(range(len(max_lyap)), max_lyap, color=colors, alpha=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    fs = max(5, 8 - len(names) // 10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("/", "\n") for n in names],
                       fontsize=fs, rotation=45, ha="right")
    ax.set_title("Max Lyapunov exponent per layer\n(blue=FNN  red=Attn)")
    ax.set_ylabel("Max exponent"); ax.grid(True, alpha=0.3)

    # --- 下段: 混同行列 ---
    if has_conf:
        ax = fig.add_subplot(n_rows, 3, (4, 6))
        n_cls = conf_mat.shape[0]
        disp  = conf_mat.astype(float)
        rs    = disp.sum(axis=1, keepdims=True)
        disp /= np.where(rs == 0, 1, rs)
        im    = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        if label_names and n_cls <= 20:
            ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
            ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(label_names, fontsize=8)
        ax.set_title("Confusion matrix (row-normalized)")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.tight_layout()
    path = f"{save_prefix}_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[保存] {path}")
    #plt.show()

def save_csv_results(lyap_result, save_prefix="lyap_result"):
    p1 = f"{save_prefix}_spectrum.csv"
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "mean_lyap", "std_lyap"])
        for i, (m, s) in enumerate(zip(lyap_result["mean_global"],lyap_result["std_global"])):
            w.writerow([i, float(m), float(s)])
    print(f"[保存] {p1}")
    p2 = f"{save_prefix}_per_layer.csv"
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "max_lyap"])
        for name, val in zip(lyap_result["layer_names"], lyap_result["mean_per_layer_max"]):
            w.writerow([name, float(val)])
    print(f"[保存] {p2}")

# ============================================================
# 6. 引数定義
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Lyapunov spectrum of Attention+FNN model")
    # タスク
    p.add_argument("--task", type=str, default="mnist", choices=["mnist","fashion_mnist","cifar10","cifar100","sin","csv"])
    p.add_argument("--data_root",  type=str, default="./data")
    p.add_argument("--data_path",  type=str, default="data.csv")
    p.add_argument("--target_col", type=int, default=-1)
    p.add_argument("--max_train",  type=int, default=None, help="学習サンプル上限 (None=全件)")
    p.add_argument("--max_test",   type=int, default=None)
    # モデル構造
    p.add_argument("--N",           type=int,   default=None, help="トークン次元 (None→自動)")
    p.add_argument("--M",           type=int,   default=None, help="トークン数  (None→自動)")
    p.add_argument("--n_blocks",    type=int,   default=4)
    p.add_argument("--attn_layers", type=int,   default=3, help="1 ブロックあたりの Attention 層数")
    p.add_argument("--fnn_layers",  type=int,   default=1, help="1 ブロックあたりの FNN 層数")
    p.add_argument("--residual",    action="store_true")
    p.add_argument("--beta",        type=float, default=2.0)
    p.add_argument("--dropout",     type=float, default=0.1)
    # 学習
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--n_samples",  type=int,   default=500)

    # 出力するリャプノフ指数の数
    p.add_argument("--lyap_samples", type=int, default=5)
    # 出力
    p.add_argument("--out_prefix", type=str, default="lyap_result")
    p.add_argument("--device",     type=str, default="cuda")
    return p.parse_args()

def makesuffix(args):
    s=f"{args.task}_n{args.n_blocks}_at{args.attn_layers}_fnn{args.fnn_layers}_beta{args.beta}_dr{args.dropout}"
    if(args.residual):
        s+="_res"
    return s
    
# ============================================================
# 7. メイン
# ============================================================
def exec(args,device):
    # N, M の決定
    dN, dM = TASK_DEFAULTS[args.task]
    N = args.N if args.N is not None else dN
    M = args.M if args.M is not None else dM
    suf=makesuffix(args)
    print("=" * 62)
    print(f"  タスク       : {args.task}")
    print(f"  N (tok dim)  : {N}   M (tok num): {M}")
    print(f"  ブロック構成 : {args.n_blocks} blocks × "
          f"(FNN×{args.fnn_layers} + Attn×{args.attn_layers})")
    print(f"  デバイス     : {device}")
    print("=" * 62)

    print("\n[1] データ準備...")
    is_image = args.task in IMAGE_DATASETS
    conf_mat    = None
    label_names = LABEL_NAMES.get(args.task)
    n_classes   = None
    img_flat    = None

    if is_image:
        train_ds, test_ds, n_classes, img_flat = load_image_dataset(
            args.task, data_root=args.data_root,
            max_train=args.max_train, max_test=args.max_test,)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        print(f"  {args.task}: train={len(train_ds)}, test={len(test_ds)}, classes={n_classes}, img_flat={img_flat}")
    else:
        if args.task == "sin":
            dataset = make_sin_dataset(n_samples=args.n_samples,
                                       seq_len=M, n_features=N, device=device)
        else:
            dataset = make_csv_dataset(args.data_path, N, M, args.target_col, device)
        n  = len(dataset)
        sp = int(n * 0.8)
        train_ds, test_ds = torch.utils.data.random_split(dataset, [sp, n - sp])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)
        print(f"  {args.task}: train={sp}, test={n - sp}")

    print("\n[2] モデル構築...")
    model = AttentionFNNModel(
        n_input        = N,
        n_seq          = M,
        n_classes      = n_classes,
        img_flat_dim   = img_flat,
        n_blocks       = args.n_blocks,
        attn_per_block = args.attn_layers,
        fnn_per_block  = args.fnn_layers,
        residual       = args.residual,
        beta           = args.beta,
        dropout        = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数 : {n_params:,}")
    print(f"  総層数       : {args.n_blocks * (args.attn_layers + args.fnn_layers)}")
    print(f"  model input as: {model.n_input*model.n_seq}")

    print("\n[3] 学習中...")
    history = train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    if n_classes is not None:
        print("  混同行列を計算中...")
        conf_mat = compute_confusion(model, test_loader, n_classes, device)
        print(f"  最終テスト精度: {history['test_acc'][-1]*100:.2f}%")

    print(f"\n[4] リャプノフスペクトル計算 (lyap_samples={args.lyap_samples})...")
    model_cpu = model.cpu()
    # テストデータを収集
    X_list = []
    for X_b, _ in test_loader:
        X_list.append(X_b)
        if sum(x.shape[0] for x in X_list) >= args.lyap_samples * 2:
            break
    X_all      = torch.cat(X_list, dim=0)
    X_embedded = embed_samples(model_cpu, X_all.cpu())
    print(f"  埋め込み後の shape: {tuple(X_embedded.shape)}")

    lyap_result = calc_lyapunov_averaged(model_cpu, X_embedded, n_samples=args.lyap_samples)

    # サマリー表示
    mg = lyap_result["mean_global"]
    print(f"\n  最大リャプノフ指数 : {mg.max():.4f}")
    print(f"  最小リャプノフ指数 : {mg.min():.4f}")
    print(f"  正の指数の個数     : {(mg > 0).sum()} / {len(mg)}")
    sl = np.sort(mg)[::-1]
    cs = np.cumsum(sl)
    ki = np.searchsorted(-cs, 0)
    if 0 < ki < len(sl) and sl[ki] != 0:
        print(f"  Kaplan-Yorke 次元 : {ki + cs[ki-1] / abs(sl[ki]):.2f}")

    print("\n[5] 保存・可視化...")

    save_csv_results(lyap_result, save_prefix=args.out_prefix+suf)
    plot_results(history, lyap_result, task=args.task,
                 save_prefix=args.out_prefix+suf,
                 conf_mat=conf_mat, label_names=label_names)

def execs():
    args   = parse_args()
    device = torch.device(args.device)
    for n_blocks ,attn_per_block ,fnn_per_block  ,residual ,beta  ,dropout  in itertools.product(
        [1,2,4],[1,4,8],[1,4,8],[True,False],[1.4,2.0],[0.2]):
        args.n_blocks=    n_blocks       
        args.attn_layers= attn_per_block 
        args.fnn_layers=  fnn_per_block  
        args.residual=    residual       
        args.beta   =    beta           
        args.dropout =    dropout        
        exec(args,device)
    print("\n完了!")

def main():
    args   = parse_args()
    device = torch.device(args.device)
    exec(args,device)
    print("\n完了!")

if __name__ == "__main__":
    execs()
