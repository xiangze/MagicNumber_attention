"""
Experiment 2b
=============
Harder task: predict x[:, 0] XOR x[:, 1] (parity of first two) — vocab=2.
Lower learning rate to slow the transition and see staged dynamics.

Compare two architectures:
  A) full block:    attn + FFN + skip + PreLN
  B) attn-only:     attn + skip + PreLN (no FFN)

Hypothesis check: does the staircase appear in *stages* (multiple
saddles), and does any band cross zero (sign change in Lyapunov
exponent during learning)?
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, "/home/claude")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformer_block import BlockConfig, TinyTransformer
from lyapunov import lyapunov_spectrum
from staircase import staircase_summary

OUT = "."
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)

VOCAB = 2
SEQ_LEN = 6
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 4


def make_batch(B, seq_len=SEQ_LEN, vocab=VOCAB):
    x = torch.randint(0, vocab, (B, seq_len))
    target = (x[:, 0] ^ x[:, 1]) % vocab   # binary XOR
    return x, target


class FullModel(nn.Module):
    def __init__(self, cfg, n_layers, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, cfg.seq_len, cfg.d_model))
        self.core = TinyTransformer(cfg, n_layers=n_layers)
        self.head = nn.Linear(cfg.d_model, vocab)

    def embed(self, ids):
        return self.emb(ids) + self.pos

    def forward(self, ids):
        return self.head(self.core(self.embed(ids))[:, -1])


def measure_spectrum(model, probe_ids, n_samples):
    spectra = []
    with torch.no_grad():
        embs = model.embed(probe_ids[:n_samples])
    for i in range(n_samples):
        lam, _ = lyapunov_spectrum(model.core, embs[i:i+1].clone())
        spectra.append(lam)
    return np.stack(spectra, axis=0).mean(axis=0)


def train_one(label, cfg, n_layers, n_steps=600, eval_every=30):
    torch.manual_seed(2)
    np.random.seed(2)
    model = FullModel(cfg, n_layers, VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    probe_ids, _ = make_batch(8)

    log = []
    model.eval()
    lam = measure_spectrum(model, probe_ids, n_samples=4)
    log.append((0, float("nan"), float("nan"), lam))

    t0 = time.time()
    for step in range(1, n_steps + 1):
        model.train()
        ids, tgt = make_batch(128)
        logits = model(ids)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                ids_e, tgt_e = make_batch(512)
                acc = (model(ids_e).argmax(-1) == tgt_e).float().mean().item()
            lam = measure_spectrum(model, probe_ids, n_samples=4)
            log.append((step, loss.item(), acc, lam))
            print(f"  [{label}] step {step:4d}  loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"λmax {lam.max():+.3f}  λmin {lam.min():+.3f}")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")

    steps = np.array([s for s, _, _, _ in log])
    losses = np.array([l for _, l, _, _ in log])
    accs = np.array([a for _, _, a, _ in log])
    specs = np.stack([sp for _, _, _, sp in log], axis=0)
    return dict(label=label, steps=steps, losses=losses, accs=accs, spectra=specs)


def plot_run(run, fname_prefix):
    steps, specs = run["steps"], run["spectra"]
    n = specs.shape[1]

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(specs.T, aspect="auto", origin="upper",
                   extent=[steps[0], steps[-1], n, 0],
                   cmap="RdBu_r",
                   vmin=-np.abs(specs).max(), vmax=np.abs(specs).max())
    ax.set_xlabel("training step"); ax.set_ylabel("rank (0=top/largest)")
    ax.set_title(f"{run['label']} — spectrum heatmap")
    fig.colorbar(im, ax=ax, label=r"$\lambda$")
    fig.tight_layout(); fig.savefig(f"{fname_prefix}_heatmap.png", dpi=140); plt.close(fig)

    # Snapshots
    fig, ax = plt.subplots(figsize=(9, 5))
    idxs = np.linspace(0, len(steps) - 1, 8).astype(int)
    cmap = plt.get_cmap("viridis")
    for j, idx in enumerate(idxs):
        ax.plot(specs[idx], color=cmap(j / max(1, len(idxs)-1)),
                lw=1.5, label=f"step {steps[idx]}")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("rank"); ax.set_ylabel(r"$\lambda$")
    ax.set_title(f"{run['label']} — snapshots")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{fname_prefix}_snapshots.png", dpi=140); plt.close(fig)

    # Bands
    k = max(3, n // 10)
    top = np.sort(specs, axis=1)[:, -k:].mean(axis=1)
    mid = np.sort(specs, axis=1)[:, (n//2 - k//2):(n//2 + k//2)].mean(axis=1)
    bot = np.sort(specs, axis=1)[:, :k].mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, top, "C3-", label=f"top-{k}")
    ax.plot(steps, mid, "C7-", label=f"mid-{k}")
    ax.plot(steps, bot, "C0-", label=f"bottom-{k}")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    # overlay loss for context
    ax2 = ax.twinx()
    valid = ~np.isnan(run["losses"])
    ax2.plot(steps[valid], run["losses"][valid], "k--", alpha=0.3, label="loss")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\lambda$"); ax2.set_ylabel("loss")
    ax.set_title(f"{run['label']} — bands")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{fname_prefix}_bands.png", dpi=140); plt.close(fig)


def main():
    cfg_full = BlockConfig(d_model=D_MODEL, n_heads=N_HEADS, d_ff=2*D_MODEL,
                           seq_len=SEQ_LEN, use_skip=True, use_ffn=True, ln_style="pre")
    cfg_attn = BlockConfig(d_model=D_MODEL, n_heads=N_HEADS, d_ff=2*D_MODEL,
                           seq_len=SEQ_LEN, use_skip=True, use_ffn=False, ln_style="pre")

    print("=== A: full block ===")
    run_full = train_one("full", cfg_full, n_layers=N_LAYERS, n_steps=600)
    plot_run(run_full, f"{OUT}/exp2b_full")

    print("\n=== B: attention-only ===")
    run_attn = train_one("attn-only", cfg_attn, n_layers=N_LAYERS, n_steps=600)
    plot_run(run_attn, f"{OUT}/exp2b_attn")

    np.savez(f"{OUT}/exp2b.npz",
             full_steps=run_full["steps"], full_spec=run_full["spectra"],
             full_loss=run_full["losses"], full_acc=run_full["accs"],
             attn_steps=run_attn["steps"], attn_spec=run_attn["spectra"],
             attn_loss=run_attn["losses"], attn_acc=run_attn["accs"])

    # combined comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, run in zip(axes, [run_full, run_attn]):
        specs = run["spectra"]; n = specs.shape[1]; k = max(3, n // 10)
        top = np.sort(specs, axis=1)[:, -k:].mean(axis=1)
        bot = np.sort(specs, axis=1)[:, :k].mean(axis=1)
        ax.plot(run["steps"], top, "C3-", label=f"top-{k}")
        ax.plot(run["steps"], bot, "C0-", label=f"bottom-{k}")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(run["label"])
        ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\lambda$")
    fig.suptitle("Spectrum band dynamics: full vs attention-only")
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2b_compare.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
