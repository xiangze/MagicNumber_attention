"""
ALBERTカオス実験スクリプト
=========================
以下の3実験を実装:
  Exp1: ランダムラベル訓練 → 訓練誤差→0, 汎化誤差→最大, λ<0 の確認
  Exp2: Token Collapse誘導 → softmax温度を下げてλ≪0を確認
  Exp3: Edge of Chaos探索 → 学習率スキャンでλ≈0となるepochと汎化誤差の関係

依存ライブラリ:
  pip install torch transformers datasets numpy matplotlib tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from transformers import AlbertForSequenceClassification, AlbertTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from datasets import load_dataset

# -------------------------------------------------------------------
# 共通ユーティリティ
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "albert-base-v2"
NUM_LABELS = 2
BATCH_SIZE = 16
MAX_LEN = 64

def load_sst2_tokenized(tokenizer, n_samples=2000):
    """SST-2から小さなサブセットを取得してトークナイズ"""
    dataset = load_dataset("glue", "sst2", split="train")
    dataset = dataset.shuffle(seed=42).select(range(n_samples))

    input_ids, attention_masks, labels = [], [], []
    for item in dataset:
        enc = tokenizer(
            item["sentence"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(enc["input_ids"].squeeze())
        attention_masks.append(enc["attention_mask"].squeeze())
        labels.append(item["label"])

    return (
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.tensor(labels, dtype=torch.long),
    )


def make_dataloaders(input_ids, attention_masks, labels, val_ratio=0.2):
    """DataLoaderを訓練/検証に分割して返す"""
    dataset = TensorDataset(input_ids, attention_masks, labels)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    return train_loader, val_loader


# -------------------------------------------------------------------
# Lyapunov指数の近似計算
# -------------------------------------------------------------------
# 実装方針: モデルのforward passをRNNとみなし、
# 各Transformerブロックを「離散写像 f_l」として扱う。
# 入力xに微小摂動εを加え、各層通過後の発散率を測定する。
# λ ≈ (1/L) * Σ_l log(||f_l(x+ε) - f_l(x)|| / ||ε||)

def estimate_lyapunov(model, input_ids, attention_mask, epsilon=1e-4, layer_indices=None):
    """
    ALBERTの各Transformerブロック出力でのLyapunov指数近似。
    ALBERTは weight sharing のため実質1ブロックの反復写像になっている点に注意。

    Returns:
        float: 近似Lyapunov指数 λ
    """
    model.eval()
    model.to(DEVICE)
    input_ids      = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        # Embedding層の出力を取得
        emb_output = model.albert.embeddings(input_ids)  # (B, T, H)

        # 微小摂動を加えた入力を作成
        noise = torch.randn_like(emb_output) * epsilon
        emb_perturbed = emb_output + noise

        log_expansion_rates = []
        hidden      = emb_output
        hidden_pert = emb_perturbed

        # ALBERTのEncoder層を順次通過（weight tying のため同一ブロックが繰り返される）
        encoder = model.albert.encoder
        for layer_idx in range(encoder.config.num_hidden_layers):
            layer = encoder.albert_layer_groups[0].albert_layers[0]

            # 拡張マスクを作成
            ext_mask = model.albert.get_extended_attention_mask(
                attention_mask, input_ids.shape
            )

            out_orig = layer(hidden,      ext_mask)[0]
            out_pert = layer(hidden_pert, ext_mask)[0]

            delta_before = (hidden_pert - hidden).norm(dim=-1).mean().item()
            delta_after  = (out_pert - out_orig).norm(dim=-1).mean().item()

            if delta_before > 1e-10:
                rate = np.log(delta_after / delta_before)
                log_expansion_rates.append(rate)

            # 摂動を再正規化（QR法の簡易版）
            diff = out_pert - out_orig
            diff = diff * (epsilon / (diff.norm(dim=-1, keepdim=True) + 1e-10))
            hidden      = out_orig
            hidden_pert = out_orig + diff

    return float(np.mean(log_expansion_rates)) if log_expansion_rates else float("nan")


# -------------------------------------------------------------------
# Exp1: ランダムラベル訓練
# -------------------------------------------------------------------

def experiment1_random_labels(n_epochs=5, n_samples=1000):
    """
    ランダムラベルでALBERTを訓練し、
    訓練誤差・汎化誤差・Lyapunov指数の推移を記録する。
    """
    print("\n" + "="*60)
    print("Exp1: Random Label Training")
    print("="*60)

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    input_ids, attention_masks, true_labels = load_sst2_tokenized(tokenizer, n_samples)

    # ランダムラベルに置換
    random_labels = torch.randint(0, NUM_LABELS, (len(true_labels),))
    print(f"Label agreement with true labels: "
          f"{(random_labels == true_labels).float().mean():.2%} (expected ~50%)")

    train_loader, val_loader = make_dataloaders(input_ids, attention_masks, random_labels)
    # 評価には真のラベルを使う（汎化誤差を測定するため）
    _, val_loader_true = make_dataloaders(input_ids, attention_masks, true_labels)

    model = AlbertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    history = {"train_loss": [], "val_acc_random": [], "val_acc_true": [], "lyapunov": []}

    for epoch in range(n_epochs):
        # 訓練
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            ids, mask, lbl = [b.to(DEVICE) for b in batch]
            outputs = model(input_ids=ids, attention_mask=mask, labels=lbl)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ランダムラベルへの適合率（訓練精度に相当）
        val_acc_rand = evaluate_accuracy(model, val_loader)
        # 真ラベルへの精度（汎化誤差の代理）
        val_acc_true = evaluate_accuracy(model, val_loader_true)

        # Lyapunov指数推定（最初のバッチで）
        sample_ids, sample_mask, _ = next(iter(val_loader))
        lya = estimate_lyapunov(model, sample_ids[:4], sample_mask[:4])

        history["train_loss"].append(avg_loss)
        history["val_acc_random"].append(val_acc_rand)
        history["val_acc_true"].append(val_acc_true)
        history["lyapunov"].append(lya)

        print(f"  Loss={avg_loss:.4f} | Acc(random_lbl)={val_acc_rand:.3f} "
              f"| Acc(true_lbl)={val_acc_true:.3f} | λ={lya:.4f}")

    plot_experiment1(history, n_epochs)
    return history


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, mask, lbl in dataloader:
            ids, mask, lbl = ids.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == lbl).sum().item()
            total   += len(lbl)
    return correct / total


def plot_experiment1(history, n_epochs):
    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o")
    axes[0].set_title("Training Loss (random labels)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, history["val_acc_random"], "r-o", label="random label acc")
    axes[1].plot(epochs, history["val_acc_true"],   "g-o", label="true label acc")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="chance level")
    axes[1].set_title("Accuracy: Random vs True Labels")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["lyapunov"], "m-o")
    axes[2].axhline(0.0, color="gray", linestyle="--", label="λ=0 (Edge of Chaos)")
    axes[2].set_title("Lyapunov Exponent λ")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("λ"); axes[2].legend()

    plt.suptitle("Exp1: Random Label Training\n"
                 "Hypothesis: λ<0 (stable) while generalization error remains high",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("exp1_random_labels.png", dpi=120, bbox_inches="tight")
    print("  → Saved: exp1_random_labels.png")
    plt.close()


# -------------------------------------------------------------------
# Exp2: Token Collapse の誘導（softmax温度操作）
# -------------------------------------------------------------------

class PatchedAlbertAttention(nn.Module):
    """
    元のAlbert Attention をラップし、softmaxに温度パラメータを適用する。
    温度 T→0 で softmax が鋭くなり → 特定トークンへの注目集中 → rank collapse
    温度 T→∞ で softmax が一様になり → token uniformity → rank collapse
    （両端で崩壊するが経路は異なる）
    """
    def __init__(self, original_attention, temperature=1.0):
        super().__init__()
        self.attn = original_attention
        self.temperature = temperature

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False):
        # ALBERTのAttentionクラスの内部に温度を注入するためにフックを使う
        original_forward = self.attn.attention.query.__class__.forward

        def scaled_forward(self_inner, x):
            return original_forward(self_inner, x) / self.temperature

        # モンキーパッチ（実験用の近似的手法）
        # 実際にはAttention scoreにT^{-1}を掛けるため、Query出力をスケール
        with TemperaturePatch(self.attn, self.temperature):
            return self.attn(hidden_states, attention_mask, head_mask, output_attentions)


class TemperaturePatch:
    """AttentionのQuery出力をscaleするcontext manager"""
    def __init__(self, attn_module, temperature):
        self.attn = attn_module
        self.T = temperature
        self._original_query_forward = None

    def __enter__(self):
        original = self.attn.attention.query.forward
        T = self.T
        def patched(x):
            return original(x) / np.sqrt(T)
        self.attn.attention.query.forward = patched
        return self

    def __exit__(self, *args):
        # パッチを元に戻す（簡易実装のためリセット）
        self.attn.attention.query.forward = (
            self.attn.attention.query.__class__.forward.__get__(
                self.attn.attention.query
            )
        )


def measure_token_diversity(model, input_ids, attention_mask):
    """
    各Transformer層の出力表現の多様性をeffective rank（近似）で測定する。
    effective rank ≈ exp(H(σ)) where H(σ)はsingular valueのエントロピー
    """
    model.eval()
    model.to(DEVICE)
    input_ids      = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    effective_ranks = []
    with torch.no_grad():
        hidden = model.albert.embeddings(input_ids)
        encoder = model.albert.encoder
        ext_mask = model.albert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        for _ in range(encoder.config.num_hidden_layers):
            layer = encoder.albert_layer_groups[0].albert_layers[0]
            hidden = layer(hidden, ext_mask)[0]

            # hidden: (B, T, H) → Tトークンの表現行列のeffective rank
            # バッチ平均
            batch_ranks = []
            for b in range(hidden.shape[0]):
                h = hidden[b].float()  # (T, H)
                try:
                    sv = torch.linalg.svdvals(h)
                    sv_norm = sv / (sv.sum() + 1e-10)
                    entropy = -(sv_norm * (sv_norm + 1e-10).log()).sum()
                    eff_rank = entropy.exp().item()
                except Exception:
                    eff_rank = float("nan")
                batch_ranks.append(eff_rank)
            effective_ranks.append(np.nanmean(batch_ranks))

    return effective_ranks


def experiment2_token_collapse(temperatures=None, n_samples=200):
    """
    softmax温度を変えてToken Collapseを誘導し、
    Effective Rank と Lyapunov指数への影響を確認する。
    """
    print("\n" + "="*60)
    print("Exp2: Token Collapse via Temperature Manipulation")
    print("="*60)

    if temperatures is None:
        temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    input_ids, attention_masks, _ = load_sst2_tokenized(tokenizer, n_samples)
    # 少量サンプルで実験
    input_ids      = input_ids[:8]
    attention_masks = attention_masks[:8]

    results = {"temperature": [], "eff_rank_mean": [], "eff_rank_last": [], "lyapunov": []}

    for T in temperatures:
        print(f"  Temperature T={T:.3f} ...", end=" ")

        # モデルを毎回クリーンに読み込む
        model = AlbertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)

        # Query出力をスケールしてsoftmax温度を模擬
        original_query_forward = {}
        for i, layer_group in enumerate(model.albert.encoder.albert_layer_groups):
            for j, layer in enumerate(layer_group.albert_layers):
                orig_fwd = layer.attention.attention.query.forward
                scale = 1.0 / np.sqrt(T)
                def make_patched(orig, s):
                    def patched(x): return orig(x) * s
                    return patched
                layer.attention.attention.query.forward = make_patched(orig_fwd, scale)

        eff_ranks = measure_token_diversity(model, input_ids, attention_masks)
        lya = estimate_lyapunov(model, input_ids[:4], attention_masks[:4])

        mean_rank = np.mean(eff_ranks)
        last_rank = eff_ranks[-1] if eff_ranks else float("nan")

        results["temperature"].append(T)
        results["eff_rank_mean"].append(mean_rank)
        results["eff_rank_last"].append(last_rank)
        results["lyapunov"].append(lya)

        print(f"eff_rank(mean)={mean_rank:.2f} | eff_rank(last_layer)={last_rank:.2f} | λ={lya:.4f}")

    plot_experiment2(results)
    return results


def plot_experiment2(results):
    T_vals = results["temperature"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogx(T_vals, results["eff_rank_mean"], "b-o", label="mean (all layers)")
    axes[0].semilogx(T_vals, results["eff_rank_last"], "r-s", label="last layer")
    axes[0].axvline(1.0, color="gray", linestyle="--", label="T=1 (baseline)")
    axes[0].set_title("Effective Rank vs Temperature")
    axes[0].set_xlabel("Temperature (log scale)"); axes[0].set_ylabel("Effective Rank")
    axes[0].legend()

    axes[1].semilogx(T_vals, results["lyapunov"], "m-o")
    axes[1].axhline(0.0, color="gray", linestyle="--", label="λ=0")
    axes[1].axvline(1.0, color="gray", linestyle=":")
    axes[1].set_title("Lyapunov Exponent vs Temperature")
    axes[1].set_xlabel("Temperature (log scale)"); axes[1].set_ylabel("λ")
    axes[1].legend()

    plt.suptitle("Exp2: Token Collapse via Softmax Temperature\n"
                 "Hypothesis: Low T → rank collapse → λ≪0",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("exp2_token_collapse.png", dpi=120, bbox_inches="tight")
    print("  → Saved: exp2_token_collapse.png")
    plt.close()


# -------------------------------------------------------------------
# Exp3: Edge of Chaos 探索（学習率スキャン）
# -------------------------------------------------------------------

def experiment3_edge_of_chaos(lr_list=None, n_epochs=3, n_samples=1000):
    """
    学習率を変えてトレーニングし、
    「λ≈0のedge of chaos epoch」と「汎化誤差」の対応を確認する。
    反例の探索：λ≈0でも汎化誤差が高い場合があるかを調べる。
    """
    print("\n" + "="*60)
    print("Exp3: Edge of Chaos Search via Learning Rate Scan")
    print("="*60)

    if lr_list is None:
        lr_list = [1e-6, 5e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    input_ids, attention_masks, true_labels = load_sst2_tokenized(tokenizer, n_samples)
    train_loader, val_loader = make_dataloaders(input_ids, attention_masks, true_labels)
    sample_ids, sample_mask, _ = next(iter(val_loader))

    all_results = {}

    for lr in lr_list:
        print(f"\n  Learning Rate = {lr:.0e}")
        model = AlbertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        epoch_records = []
        for epoch in range(n_epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"  LR={lr:.0e} Epoch {epoch+1}", leave=False):
                ids, mask, lbl = [b.to(DEVICE) for b in batch]
                loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_acc = evaluate_accuracy(model, val_loader)
            lya = estimate_lyapunov(model, sample_ids[:4], sample_mask[:4])
            epoch_records.append({"epoch": epoch + 1, "val_acc": val_acc, "lyapunov": lya})
            print(f"    Epoch {epoch+1}: val_acc={val_acc:.3f} | λ={lya:.4f}")

        all_results[lr] = epoch_records

    plot_experiment3(all_results, lr_list, n_epochs)
    analyze_edge_of_chaos_counterexamples(all_results)
    return all_results


def analyze_edge_of_chaos_counterexamples(all_results, lambda_threshold=0.05):
    """
    「λ≈0 かつ 汎化誤差が高い」という反例候補を検出する。
    """
    print("\n  --- Counterexample Analysis (λ≈0 but high generalization error) ---")
    found = False
    for lr, records in all_results.items():
        for rec in records:
            if abs(rec["lyapunov"]) < lambda_threshold and rec["val_acc"] < 0.65:
                print(f"  *** Counterexample candidate: LR={lr:.0e}, "
                      f"Epoch={rec['epoch']}, λ={rec['lyapunov']:.4f}, "
                      f"val_acc={rec['val_acc']:.3f} (low accuracy at edge!)")
                found = True
    if not found:
        print("  No clear counterexamples found with current settings. "
              "Try random labels + edge of chaos scan (Exp1 × Exp3 combination).")


def plot_experiment3(all_results, lr_list, n_epochs):
    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / len(lr_list)) for i in range(len(lr_list))]

    for (lr, records), color in zip(all_results.items(), colors):
        lyas = [r["lyapunov"] for r in records]
        accs = [r["val_acc"]  for r in records]
        label = f"lr={lr:.0e}"
        axes[0].plot(epochs, lyas, "-o", color=color, label=label)
        axes[1].plot(epochs, accs, "-o", color=color, label=label)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=2, label="λ=0 (Edge)")
    axes[0].set_title("Lyapunov Exponent λ per Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("λ"); axes[0].legend(fontsize=7)

    axes[1].axhline(0.5, color="gray", linestyle="--", label="chance level")
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(fontsize=7)

    plt.suptitle("Exp3: Edge of Chaos Search via Learning Rate Scan\n"
                 "Key question: Is λ≈0 sufficient for good generalization?",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig("exp3_edge_of_chaos.png", dpi=120, bbox_inches="tight")
    print("  → Saved: exp3_edge_of_chaos.png")
    plt.close()


# -------------------------------------------------------------------
# Exp1 × Exp3 複合実験：ランダムラベル + 学習率スキャン
# （最重要反例：λ≈0 かつ 汎化誤差=最大 を意図的に作る）
# -------------------------------------------------------------------

def experiment_combined_counterexample(lr_list=None, n_epochs=3, n_samples=500):
    """
    ランダムラベルで訓練しつつ学習率を調整することで、
    Edge of Chaos (λ≈0) に近い状態で汎化誤差が最大になる反例を作る。
    これがEdge of Chaos論文への最も直接的な反例。
    """
    print("\n" + "="*60)
    print("Combined: Random Labels + LR Scan (Direct Counterexample)")
    print("="*60)

    if lr_list is None:
        lr_list = [1e-5, 3e-5, 1e-4]

    tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
    input_ids, attention_masks, true_labels = load_sst2_tokenized(tokenizer, n_samples)
    random_labels = torch.randint(0, NUM_LABELS, (len(true_labels),))

    # 訓練はランダムラベル、評価は真ラベル
    train_loader, _ = make_dataloaders(input_ids, attention_masks, random_labels)
    _, val_loader   = make_dataloaders(input_ids, attention_masks, true_labels)
    sample_ids, sample_mask, _ = next(iter(val_loader))

    scatter_data = {"lyapunov": [], "val_acc": [], "lr": []}

    for lr in lr_list:
        print(f"\n  LR={lr:.0e} (random labels)")
        model = AlbertForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False):
                ids, mask, lbl = [b.to(DEVICE) for b in batch]
                loss = model(input_ids=ids, attention_mask=mask, labels=lbl).loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            val_acc = evaluate_accuracy(model, val_loader)
            lya = estimate_lyapunov(model, sample_ids[:4], sample_mask[:4])
            scatter_data["lyapunov"].append(lya)
            scatter_data["val_acc"].append(val_acc)
            scatter_data["lr"].append(lr)
            print(f"    Epoch {epoch+1}: λ={lya:.4f} | val_acc(true)={val_acc:.3f}")

    # λ と val_acc の散布図
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("Set1")
    for i, lr in enumerate(lr_list):
        mask = [s == lr for s in scatter_data["lr"]]
        xs = [scatter_data["lyapunov"][j] for j in range(len(mask)) if mask[j]]
        ys = [scatter_data["val_acc"][j]  for j in range(len(mask)) if mask[j]]
        ax.scatter(xs, ys, label=f"lr={lr:.0e}", s=80, color=cmap(i))

    ax.axvline(0.0, color="black", linestyle="--", label="λ=0 (Edge of Chaos)")
    ax.axhline(0.5, color="gray",  linestyle="--", label="chance level")
    ax.set_xlabel("Lyapunov Exponent λ")
    ax.set_ylabel("Validation Accuracy (true labels)")
    ax.set_title("Direct Counterexample: λ≈0 with Worst Generalization\n"
                 "(Trained on random labels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("exp_combined_counterexample.png", dpi=120, bbox_inches="tight")
    print("  → Saved: exp_combined_counterexample.png")
    plt.close()

    return scatter_data


# -------------------------------------------------------------------
# メイン実行
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("ALBERT Chaos Experiments")
    print(f"Device: {DEVICE}")
    print("NOTE: Full runs require GPU. Reduce n_samples/n_epochs for quick testing.\n")

    # --- クイックテスト用設定（本番は値を増やす） ---
    QUICK = True  # Falseにすると本格実験

    if QUICK:
        print("[QUICK MODE] Reduced samples/epochs for debugging")
        h1 = experiment1_random_labels(n_epochs=2, n_samples=200)
        r2 = experiment2_token_collapse(temperatures=[0.1, 0.5, 1.0, 5.0], n_samples=100)
        r3 = experiment3_edge_of_chaos(lr_list=[1e-5, 1e-4, 1e-3], n_epochs=2, n_samples=200)
        rc = experiment_combined_counterexample(lr_list=[1e-5, 1e-4], n_epochs=2, n_samples=200)
    else:
        print("[FULL MODE]")
        h1 = experiment1_random_labels(n_epochs=5, n_samples=2000)
        r2 = experiment2_token_collapse()
        r3 = experiment3_edge_of_chaos()
        rc = experiment_combined_counterexample()

    print("\nAll experiments complete. Check PNG files for results.")
