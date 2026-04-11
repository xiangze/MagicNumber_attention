"""
Flat Minimum / Zero-Eigenvalue Generalization Metric 実験
=========================================================

【概念的枠組み】
  Edge of Chaos (λ_Lyapunov ≈ 0) の拡張として、
  損失景観のHessian固有値スペクトルを用いた汎化性能分類:

    鋭い極小 (Sharp minimum):
      → Hessianの大固有値が多い, ゼロ固有値が少ない
      → 汎化誤差が高い傾向 (Keskar et al. 2017)

    平坦な極小 (Flat minimum):
      → 大多数の固有値がゼロ付近 (near-zero)
      → 汎化誤差が低い傾向 (Hochreiter & Schmidhuber 1997)

  提案指標: φ(θ) = #{λ_i : |λ_i| < τ} / P  (τ: 閾値, P: パラメータ数)
  これを「Flatness Ratio」と呼ぶ。

【3つの数値実験】
  Exp A: 正規訓練 vs ランダムラベル訓練 → flatness ratioと汎化誤差の対比
  Exp B: バッチサイズ変化 → 大バッチ=sharp, 小バッチ=flat の古典的結果の再現
  Exp C: 学習率 × flatness ratio × Lyapunov指数 の三角関係の可視化

【Hessian固有値の計算戦略】
  全パラメータのHessianは P×P 行列 (Pは数百万) で直接計算不可能。
  → ランチョス法 (Lanczos) + Hessian-Vector Product (HVP) で
    上位・下位 k 個の固有値を近似計算。
  → HVPは二重逆伝播で O(P) で計算可能。

依存ライブラリ:
  pip install torch transformers datasets numpy matplotlib scipy tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse.linalg import LinearOperator, eigsh
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tqdm import tqdm

from transformers import AlbertForSequenceClassification, AlbertTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset
import plots_plotly as pl
from util import dprint, banner
# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "albert-base-v2"
NUM_LABELS = 2
MAX_LEN    = 64
ZERO_EIGENVALUE_THRESHOLD = 0.01   # |λ| < τ をゼロ固有値とみなす
N_LANCZOS_EIGENVALUES     = 50     # Lanczosで推定する固有値数
N_HESSIAN_SAMPLES         = 32     # HVP計算に使うバッチサイズ


# ─────────────────────────────────────────────
# データ
# ─────────────────────────────────────────────

def load_sst2(tokenizer, n_samples=1000, seed=42):
    ds = load_dataset("glue", "sst2", split="train").shuffle(seed=seed).select(range(n_samples))
    input_ids, masks, labels = [], [], []
    for item in ds:
        enc = tokenizer(item["sentence"], max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        input_ids.append(enc["input_ids"].squeeze())
        masks.append(enc["attention_mask"].squeeze())
        labels.append(item["label"])
    return (torch.stack(input_ids),
            torch.stack(masks),
            torch.tensor(labels, dtype=torch.long))


def make_loaders(ids, masks, labels, batch_size=16, val_ratio=0.2, seed=42):
    ds = TensorDataset(ids, masks, labels)
    n_val = int(len(ds) * val_ratio)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size))


# ─────────────────────────────────────────────
# Hessian-Vector Product (HVP)
# ─────────────────────────────────────────────

def hessian_vector_product(loss, params, vector):
    """
    H・v を二重逆伝播で計算する。
    params: list of parameters (requires_grad=True)
    vector: list of tensors, params と同形
    Returns: list of tensors (H・v の各パラメータ成分)
    """
    grads = torch.autograd.grad(loss, params,
                                create_graph=True, retain_graph=True,
                                allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, params)]

    # grad と v の内積 (スカラー)
    grad_v = sum((g * v).sum() for g, v in zip(grads, vector))

    # 二階微分
    hvp = torch.autograd.grad(grad_v, params,
                               retain_graph=True, allow_unused=True)
    return [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, params)]

def get_loss_on_batch(model, batch, device):
    """評価用バッチで損失を計算（グラフ保持）"""
    ids, mask, lbl = [b.to(device) for b in batch]
    return model(input_ids=ids, attention_mask=mask, labels=lbl).loss

# ─────────────────────────────────────────────
# Lanczosによる固有値近似
# ─────────────────────────────────────────────
def lanczos_eigenvalues(model, dataloader, n_eigs=50,
                        n_iter=None, device=DEVICE,
                        param_filter=None):
    """
    Lanczos法でHessianの固有値スペクトルを近似。

    param_filter: 対象パラメータのフィルタ関数 (e.g., Attentionのみ)
                  None の場合は全パラメータ対象

    Returns:
        eigenvalues: np.ndarray (n_eigs,)
        flatness_ratio: float
        effective_rank: float  (固有値分布のShannon entropy版)
    """
    model.eval()
    model.to(device)

    # 対象パラメータの選定
    if param_filter is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [p for p, n in zip(model.parameters(),
                                    [n for n, _ in model.named_parameters()])
                  if p.requires_grad and param_filter(n)]

    P = sum(p.numel() for p in params)
    if n_iter is None:
        n_iter = min(n_eigs * 3, P, 200)  # メモリ節約

    print(f"    [Lanczos] P={P:,} | n_eigs={n_eigs} | n_iter={n_iter}")

    # HVP の線形演算子
    # 複数バッチで平均して安定化
    batch = next(iter(dataloader))
    loss = get_loss_on_batch(model, batch, device)

    def matvec(v_flat):
        """v_flat: np.ndarray (P,) → H・v (np.ndarray P,)"""
        v_flat_t = torch.tensor(v_flat, dtype=torch.float32, device=device)

        # フラットベクトルをパラメータ形状に分割
        v_split = []
        offset = 0
        for p in params:
            n = p.numel()
            v_split.append(v_flat_t[offset:offset + n].view_as(p))
            offset += n

        with torch.enable_grad():
            # lossを再計算（グラフ保持のため毎回必要）
            _loss = get_loss_on_batch(model, batch, device)
            hvp = hessian_vector_product(_loss, params, v_split)

        result = torch.cat([h.detach().reshape(-1) for h in hvp])
        return result.cpu().numpy().astype(np.float64)

    # scipy LinearOperator として包む
    H_op = LinearOperator((P, P), matvec=matvec, dtype=np.float64)

    # 上位 n_eigs/2 個と下位 n_eigs/2 個を取得
    k = min(n_eigs, P - 2)
    try:
        # 最大固有値側
        eigs_max, _ = eigsh(H_op, k=k // 2 + 1, which="LM", tol=1e-3, maxiter=300)
        # ゼロ付近
        eigs_sm, _  = eigsh(H_op, k=k // 2 + 1, which="SM", tol=1e-3, maxiter=300)
        eigenvalues = np.concatenate([eigs_max, eigs_sm])
    except Exception as e:
        print(f"    [Lanczos] eigsh failed ({e}), using fallback power iteration")
        eigenvalues = power_iteration_eigenvalues(model, batch, params, k, device)

    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # 絶対値降順

    # Flatness Ratio
    n_near_zero = np.sum(np.abs(eigenvalues) < ZERO_EIGENVALUE_THRESHOLD)
    flatness_ratio = n_near_zero / len(eigenvalues)

    # Effective Rank（固有値エントロピー版）
    ev_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
    entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-10))
    effective_rank_ev = np.exp(entropy)

    return eigenvalues, flatness_ratio, effective_rank_ev


def power_iteration_eigenvalues(model, batch, params, k, device, n_power_iter=50):
    """
    Lanczosが失敗した場合のフォールバック：
    パワーイテレーションで最大固有値のみ推定。
    """
    eigenvalues = []
    with torch.enable_grad():
        for _ in range(min(k, 10)):
            v = [torch.randn_like(p) for p in params]
            # 正規化
            norm = sum((vi ** 2).sum() for vi in v).sqrt()
            v = [vi / norm for vi in v]

            for _ in range(n_power_iter):
                loss = get_loss_on_batch(model, batch, device)
                hvp = hessian_vector_product(loss, params, v)
                norm = sum((h ** 2).sum() for h in hvp).sqrt()
                v = [h / (norm + 1e-10) for h in hvp]

            loss = get_loss_on_batch(model, batch, device)
            hvp = hessian_vector_product(loss, params, v)
            rayleigh = sum((h * vi).sum() for h, vi in zip(hvp, v))
            eigenvalues.append(rayleigh.item())

    return np.array(eigenvalues)

# ─────────────────────────────────────────────
# Lyapunov指数（前回実験から流用・簡略版）
# ─────────────────────────────────────────────
def estimate_lyapunov(model, input_ids, attention_mask, epsilon=1e-4):
    model.eval()
    model.to(DEVICE)
    input_ids      = input_ids[:4].to(DEVICE)
    attention_mask = attention_mask[:4].to(DEVICE)

    with torch.no_grad():
        emb = model.albert.embeddings(input_ids)
        noise = torch.randn_like(emb) * epsilon
        emb_p = emb + noise

        log_rates = []
        h, hp = emb, emb_p
        encoder = model.albert.encoder
        hidden = encoder.embedding_hidden_mapping_in(emb)
        hidden_pert = encoder.embedding_hidden_mapping_in(emb_p)

        ext_mask = model.albert.get_extended_attention_mask(attention_mask, input_ids.shape)

        for _ in range(encoder.config.num_hidden_layers):
            layer = encoder.albert_layer_groups[0].albert_layers[0]
            # 拡張マスクを作成
            ext_mask = model.albert.get_extended_attention_mask(attention_mask, input_ids.shape)
            try:
                out = layer(hidden,ext_mask)[0]
            except  Exception as e:
                print(e)
                print("attention_mask",attention_mask.shape, input_ids.shape)
                print("hidden",hidden.shape,"mask",ext_mask.shape)
                exit()
            outp = layer(hidden_pert, ext_mask)[0]

            d_before = (hp - h).norm(dim=-1).mean().item()
            d_after  = (outp - out).norm(dim=-1).mean().item()
            if d_before > 1e-10:
                log_rates.append(np.log(d_after / d_before + 1e-10))

            diff = outp - out
            diff = diff * (epsilon / (diff.norm(dim=-1, keepdim=True) + 1e-10))
            h, hp = out, out + diff

    return float(np.mean(log_rates)) if log_rates else float("nan")

# ─────────────────────────────────────────────
# 訓練ユーティリティ
# ─────────────────────────────────────────────

def train_model(model, train_loader, n_epochs, lr, device=DEVICE):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}", leave=False):
            ids, mask, lbl = [b.to(device) for b in batch]
            loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def evaluate(model, loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, mask, lbl in loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            preds = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            correct += (preds == lbl).sum().item()
            total += len(lbl)
    return correct / total


# ─────────────────────────────────────────────
# 結果格納
# ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    label: str
    eigenvalues: np.ndarray
    flatness_ratio: float
    effective_rank_ev: float
    lyapunov: float
    val_acc: float
    train_acc: float
    extra: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Exp A: 正規訓練 vs ランダムラベル
# ─────────────────────────────────────────────
def experiment_A(n_samples=600, n_epochs=3, lr=2e-5):
    """
    正規訓練モデルとランダムラベル訓練モデルを比較。
    仮説:
      正規訓練 → flat minimum → flatness_ratio ↑ → 汎化 ↑
      ランダム → sharp minimum or degenerate → flatness_ratio ↓ or ↑ (別の理由で)
      → 固有値分布の"形"が異なることを確認
    """
    print("\n" + "="*60)
    print("Exp A: Normal Training vs Random Labels")
    print("="*60)

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    ids, masks, true_labels = load_sst2(tokenizer, n_samples)
    rand_labels = torch.randint(0, NUM_LABELS, (len(true_labels),))

    results = []

    for label, train_labels in [("Normal",  true_labels),
                                 ("Random",  rand_labels)]:
        print(f"\n  [{label}]")
        train_loader, val_loader = make_loaders(ids, masks, train_labels)
        _, val_loader_true = make_loaders(ids, masks, true_labels)

        model = AlbertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS)
        model = train_model(model, train_loader, n_epochs, lr)

        tr_acc = evaluate(model, train_loader)
        va_acc = evaluate(model, val_loader_true)
        print(f"  train_acc={tr_acc:.3f} | val_acc(true)={va_acc:.3f}")

        sample_ids, sample_mask, _ = next(iter(val_loader_true))
        lya = estimate_lyapunov(model, sample_ids, sample_mask)

        # Hessian固有値（計算コストを抑えるため少数パラメータに絞る）
        # ALBERTのclassifier headのみを対象にして高速化
        def head_filter(name): return "classifier" in name or "pooler" in name
        eigs, flat_ratio, eff_rank = lanczos_eigenvalues(
            model, val_loader_true, n_eigs=20,
            param_filter=head_filter)

        print(f"  flatness_ratio={flat_ratio:.3f} | eff_rank_ev={eff_rank:.2f} | λ={lya:.4f}")

        results.append(ExperimentResult(
            label=label,
            eigenvalues=eigs,
            flatness_ratio=flat_ratio,
            effective_rank_ev=eff_rank,
            lyapunov=lya,
            val_acc=va_acc,
            train_acc=tr_acc,
        ))

    plot_A(results)
    return results

def plot_A(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for r in results:
        axes[0].semilogy(range(len(r.eigenvalues)), np.abs(r.eigenvalues) + 1e-10,
                         label=r.label, marker="o", markersize=3)
    axes[0].axhline(ZERO_EIGENVALUE_THRESHOLD, color="gray", linestyle="--",
                    label=f"τ={ZERO_EIGENVALUE_THRESHOLD}")
    axes[0].set(title="Hessian Eigenvalue Spectrum",
                xlabel="Index (sorted by |λ|)", ylabel="|λ| (log scale)")
    axes[0].legend()
    pl._plot_flatness_vs_acc(axes[1], results)
    axes[1].legend()
    pl._plot_lyapunov_vs_flatness(axes[2], results)
    pl._save_fig(fig, "expA_flat_minimum.png",
              "Exp A: Normal Training vs Random Label Training\n"
              "Flat minimum (high φ) should correlate with good generalization")

# ─────────────────────────────────────────────
# Exp B: バッチサイズ変化 → Sharp vs Flat
# ─────────────────────────────────────────────
def experiment_B(batch_sizes=None, n_samples=800, n_epochs=3, lr=2e-5):
    """
    Keskar et al. 2017 の「大バッチ→sharp」「小バッチ→flat」を
    flatness ratio で確認。
    ゼロ固有値数でSharp/Flatを定量化できるかの検証。
    """
    print("\n" + "="*60)
    print("Exp B: Batch Size → Sharp vs Flat Minimum")
    print("="*60)

    if batch_sizes is None:
        batch_sizes = [4, 16, 64, 256]

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    ids, masks, labels = load_sst2(tokenizer, n_samples)

    results = []
    for bs in batch_sizes:
        print(f"\n  [Batch Size = {bs}]")
        train_loader, val_loader = make_loaders(ids, masks, labels, batch_size=bs)

        model = AlbertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS)
        model = train_model(model, train_loader, n_epochs, lr)

        tr_acc = evaluate(model, train_loader)
        va_acc = evaluate(model, val_loader)
        print(f"  train_acc={tr_acc:.3f} | val_acc={va_acc:.3f}")

        sample_ids, sample_mask, _ = next(iter(val_loader))
        lya = estimate_lyapunov(model, sample_ids, sample_mask)

        def head_filter(name): return "classifier" in name or "pooler" in name
        eigs, flat_ratio, eff_rank = lanczos_eigenvalues(
            model, val_loader, n_eigs=20, param_filter=head_filter)

        print(f"  flatness_ratio={flat_ratio:.3f} | λ={lya:.4f}")

        results.append(ExperimentResult(
            label=f"BS={bs}",
            eigenvalues=eigs,
            flatness_ratio=flat_ratio,
            effective_rank_ev=eff_rank,
            lyapunov=lya,
            val_acc=va_acc,
            train_acc=tr_acc,
            extra={"batch_size": bs},
        ))

    plot_B(results)
    return results

def plot_B(results):
    bs_vals = [r.extra["batch_size"] for r in results]
    colors  = np.log10(bs_vals)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # --- (1) バッチサイズ vs Flatness Ratio ---
    axes[0].semilogx(bs_vals, [r.flatness_ratio for r in results], "b-o", markersize=8)
    for bs, fr in zip(bs_vals, [r.flatness_ratio for r in results]):
        axes[0].annotate(f"BS={bs}", (bs, fr),  textcoords="offset points", xytext=(0, 8), ha="center")
    axes[0].set(xlabel="Batch Size (log scale)", ylabel="Flatness Ratio φ(θ)",
                title="Batch Size vs Flatness Ratio\n(Keskar 2017: large BS → sharp minimum)")
    # --- (2) Flatness Ratio vs Val Acc ---
    sc = pl._plot_flatness_vs_acc(axes[1], results, c=colors, cmap="coolwarm")
    plt.colorbar(sc, ax=axes[1], label="log10(Batch Size)")
    # --- (3) Lyapunov vs Flatness Ratio ---
    sc2 = pl._plot_lyapunov_vs_flatness(axes[2], results, c=colors, cmap="coolwarm")
    pl._save_fig(fig, "expB_batch_flatness.png",
              "Exp B: Batch Size Experiment\n"
              "Verifying that zero-eigenvalue count tracks flatness")

# ─────────────────────────────────────────────
# Exp C: 3次元空間での分類地図
#   軸: (学習率, flatness_ratio, Lyapunov指数)
#   色: 汎化誤差
# ─────────────────────────────────────────────
def experiment_C(lr_list=None, n_samples=600, n_epochs=3):
    """
    学習率 × Flatness Ratio × Lyapunov指数 の3次元空間で
    汎化性能を可視化。

    主要命題の検証:
      (1) Flatness Ratio単独 vs Lyapunov単独 どちらが予測力が高いか
      (2) 両者を組み合わせると汎化誤差の予測精度が上がるか
      (3) 「λ≈0 かつ φ低」= Edge of Chaos でも汎化悪い (反例再確認)
    """
    print("\n" + "="*60)
    print("Exp C: 3D Classification Map (LR × Flatness × Lyapunov)")
    print("="*60)

    if lr_list is None:
        lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    ids, masks, labels = load_sst2(tokenizer, n_samples)
    rand_labels = torch.randint(0, NUM_LABELS, (len(labels),))

    results_normal = []
    results_random = []

    for training_type, train_labels, result_list in [
        ("Normal", labels,      results_normal),
        ("Random", rand_labels, results_random),
    ]:
        print(f"\n  === {training_type} labels ===")
        for lr in lr_list:
            print(f"  LR={lr:.0e} ...", end=" ")
            train_loader, val_loader = make_loaders(ids, masks, train_labels,
                                                    batch_size=16)
            _, val_loader_true = make_loaders(ids, masks, labels, batch_size=16)

            model = AlbertForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS)
            model = train_model(model, train_loader, n_epochs, lr)

            tr_acc = evaluate(model, train_loader)
            va_acc = evaluate(model, val_loader_true)

            sample_ids, sample_mask, _ = next(iter(val_loader_true))
            lya = estimate_lyapunov(model, sample_ids, sample_mask)

            def head_filter(name): return "classifier" in name
            eigs, flat_ratio, eff_rank = lanczos_eigenvalues(
                model, val_loader_true, n_eigs=16, param_filter=head_filter)

            print(f"φ={flat_ratio:.3f} | λ={lya:.4f} | val={va_acc:.3f}")

            result_list.append(ExperimentResult(
                label=f"{training_type}_lr{lr:.0e}",
                eigenvalues=eigs,
                flatness_ratio=flat_ratio,
                effective_rank_ev=eff_rank,
                lyapunov=lya,
                val_acc=va_acc,
                train_acc=tr_acc,
                extra={"lr": lr, "training_type": training_type},
            ))

    plot_C(results_normal, results_random)
    analyze_predictive_power(results_normal + results_random)
    return results_normal, results_random

def plot_C(results_normal, results_random):
    all_results = results_normal + results_random
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # 上段左: Normal/Random を色分けして flatness vs acc
    ax1 = fig.add_subplot(gs[0, 0])
    for res, color, marker, lbl in [
        (results_normal, "blue", "o", "Normal training"),
        (results_random, "red",  "^", "Random labels"), ]:
        pl._scatter_annotate(ax1, [r.flatness_ratio for r in res],
                          [r.val_acc for r in res],
                          [r.label  for r in res],
                          c=color, marker=marker, s=80, alpha=0.7, label=lbl)
    ax1.axhline(0.5, color="gray", linestyle="--")
    ax1.set(xlabel="Flatness Ratio φ(θ)", ylabel="Val Accuracy",
            title="Flatness Ratio vs Generalization")
    ax1.legend(fontsize=8)

    # 上段中: lyapunov vs acc
    ax2 = fig.add_subplot(gs[0, 1])
    for res, color, marker in [(results_normal, "blue", "o"), (results_random, "red", "^")]:
        pl._scatter_annotate(ax2, [r.lyapunov for r in res], [r.val_acc for r in res],
                          [r.label for r in res], c=color, marker=marker, s=80, alpha=0.7)
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.axvline(0.0, color="black", linestyle="--", label="λ=0")
    ax2.set(xlabel="Lyapunov Exponent λ", ylabel="Val Accuracy",
            title="Lyapunov λ vs Generalization\n(λ≈0 with low acc = counterexample)")
    ax2.legend(fontsize=8)

    # 上段右: φ × λ joint space
    ax3 = fig.add_subplot(gs[0, 2])
    sc = pl._scatter_annotate(ax3, [r.lyapunov for r in all_results],
                           [r.flatness_ratio for r in all_results],
                           [r.label for r in all_results],
                           c=[r.val_acc for r in all_results],
                           cmap="RdYlGn", s=120, vmin=0.4, vmax=1.0)
    plt.colorbar(sc, ax=ax3, label="Val Accuracy")
    ax3.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="λ=0")
    ax3.set(xlabel="Lyapunov Exponent λ", ylabel="Flatness Ratio φ(θ)",
            title="λ × φ joint space\n(Green=good generalization)")
    ax3.legend()

    # 下段左: 3D プロット
    ax4 = fig.add_subplot(gs[1, :2], projection="3d")
    sc3d = ax4.scatter([np.log10(r.extra["lr"]) for r in all_results],
                       [r.flatness_ratio         for r in all_results],
                       [r.lyapunov               for r in all_results],
                       c=[r.val_acc for r in all_results],
                       cmap="RdYlGn", s=100, vmin=0.4, vmax=1.0)
    plt.colorbar(sc3d, ax=ax4, label="Val Accuracy", shrink=0.6)
    ax4.set(xlabel="log10(LR)", ylabel="Flatness Ratio φ", zlabel="Lyapunov λ",
            title="3D: LR × φ × λ\n(Color = Generalization)")

    # 下段右: 予測力比較
    ax5 = fig.add_subplot(gs[1, 2])
    phi_vals = np.array([r.flatness_ratio for r in all_results])
    lya_vals = np.array([r.lyapunov       for r in all_results])
    acc_vals = np.array([r.val_acc        for r in all_results])
    metrics  = {"φ alone": phi_vals, "λ alone": lya_vals, "φ-λ\ncombined": phi_vals - lya_vals}
    corrs    = {k: abs(np.corrcoef(v, acc_vals)[0, 1]) for k, v in metrics.items()}
    bars = ax5.bar(corrs.keys(), corrs.values(), color=["steelblue", "salmon", "mediumpurple"])
    ax5.set(ylim=(0, 1), ylabel="|Pearson r| with Val Accuracy",
            title="Predictive Power of Each Metric\n(Higher |r| = better generalization predictor)")
    for bar, v in zip(bars, corrs.values()):
        ax5.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    pl._save_fig(fig, "expC_3d_map.png",
              "Exp C: 3D Classification Map of Generalization Metrics\n"
              "Testing: Flatness Ratio (φ) vs Lyapunov (λ) as generalization predictors")

def analyze_predictive_power(all_results):
    """φとλの独立性・相補性の統計的分析"""
    print("\n  --- Predictive Power Analysis ---")
    phi = np.array([r.flatness_ratio for r in all_results])
    lya = np.array([r.lyapunov       for r in all_results])
    acc = np.array([r.val_acc        for r in all_results])

    r_phi = np.corrcoef(phi, acc)[0, 1]
    r_lya = np.corrcoef(lya, acc)[0, 1]
    r_comb = np.corrcoef(phi - lya, acc)[0, 1]
    r_phi_lya = np.corrcoef(phi, lya)[0, 1]  # φとλの相関（独立性の確認）

    print(f"  Corr(φ, Val Acc) = {r_phi:+.3f}")
    print(f"  Corr(λ, Val Acc) = {r_lya:+.3f}")
    print(f"  Corr(φ-λ, Val Acc) = {r_comb:+.3f}  ← combined metric")
    print(f"  Corr(φ, λ) = {r_phi_lya:+.3f}  ← independence check")
    if abs(r_phi_lya) < 0.5:
        print("  → φ and λ are largely INDEPENDENT metrics (good: complementary)")
    else:
        print("  → φ and λ are correlated (redundant; one may suffice)")

    # 反例検出: λ≈0 かつ val_acc < 0.6
    counterexamples = [(r.label, r.lyapunov, r.flatness_ratio, r.val_acc)
                       for r in all_results
                       if abs(r.lyapunov) < 0.05 and r.val_acc < 0.6]
    if counterexamples:
        print("\n  *** Counterexamples (λ≈0 but low accuracy): ***")
        for lbl, lya_v, phi_v, acc_v in counterexamples:
            print(f"    {lbl}: λ={lya_v:.4f}, φ={phi_v:.3f}, val_acc={acc_v:.3f}")
    else:
        print("\n  No λ≈0 counterexamples found (may need more random-label runs)")

# ─────────────────────────────────────────────
# Bonus: Flatness Ratio の閾値 τ 感度分析
# ─────────────────────────────────────────────
def analyze_threshold_sensitivity(result: ExperimentResult,
                                   tau_range=None):
    """
    閾値 τ の選び方がFlatness Ratio にどう影響するかを分析。
    τ は恣意的パラメータであることの限界を示す。
    """
    if tau_range is None:
        tau_range = np.logspace(-4, 1, 50)

    flatness_ratios = []
    for tau in tau_range:
        n_zero = np.sum(np.abs(result.eigenvalues) < tau)
        flatness_ratios.append(n_zero / len(result.eigenvalues))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(tau_range, flatness_ratios, "b-o", markersize=4)
    ax.axvline(ZERO_EIGENVALUE_THRESHOLD, color="red", linestyle="--",
               label=f"default τ={ZERO_EIGENVALUE_THRESHOLD}")
    ax.set_xlabel("Threshold τ (log scale)")
    ax.set_ylabel("Flatness Ratio φ(θ)")
    ax.set_title(f"Threshold Sensitivity: '{result.label}'\n"
                 "φ depends on τ — intrinsic limitation of this metric")
    ax.legend()
    plt.tight_layout()
    plt.savefig("bonus_threshold_sensitivity.png", dpi=120, bbox_inches="tight")
    print("  → Saved: bonus_threshold_sensitivity.png")
    plt.close()

    return tau_range, flatness_ratios


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Flat Minimum / Zero-Eigenvalue Generalization Experiments")
    print(f"Device: {DEVICE}")
    print(f"Zero-eigenvalue threshold τ = {ZERO_EIGENVALUE_THRESHOLD}\n")

    QUICK = True  # False にすると本格実験

    if QUICK:
        print("[QUICK MODE]")
        res_A = experiment_A(n_samples=300, n_epochs=2, lr=2e-5)
        res_B = experiment_B(batch_sizes=[8, 64], n_samples=300, n_epochs=2)
        res_C_norm, res_C_rand = experiment_C(
            lr_list=[1e-5, 1e-4, 1e-3], n_samples=300, n_epochs=2)
        # 閾値感度分析（Exp Aの最初の結果を使用）
        if res_A:
            analyze_threshold_sensitivity(res_A[0])
    else:
        print("[FULL MODE]")
        res_A = experiment_A()
        res_B = experiment_B()
        res_C_norm, res_C_rand = experiment_C()
        if res_A:
            analyze_threshold_sensitivity(res_A[0])

    print("\nAll experiments complete.")
    print("Output files:")
    print("  expA_flat_minimum.png")
    print("  expB_batch_flatness.png")
    print("  expC_3d_map.png")
    print("  bonus_threshold_sensitivity.png")
