"""
Experiment 2
============
Hypothesis: during training, the strongly-contracting directions
(most-negative Lyapunov exponents) collapse toward zero first.
Equivalently: the staircase loses its *bottom* steps over training,
not its top steps.

Setup:
  - A small Transformer (skip + Pre-LN + FFN) trained on a synthetic
    in-context task: parity / modular copy of last-k tokens.
  - Periodically, compute the LOCAL Lyapunov spectrum on a held-out
    batch (averaged over a few inputs).
  - Track:
      * full spectrum over time
      * lambda_min (most negative)
      * lambda_max
      * spread
      * staircase metrics
      * directionally: how the *bottom-k* exponents evolve vs top-k

We add a tiny output head (linear) and minimize cross-entropy on the
predicted last token. The internal block stack remains the dynamical
system we analyze.
"""
from __future__ import annotations
import sys, os, time, json
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

# ---------------- task: induction-head style ----------------
# Each sequence: random tokens, last token is determined by simple rule:
#   target = x[:, 0]  (copy first token to be predicted at last position)
# Combined with positional info, this is solvable with attention.
VOCAB = 5
SEQ_LEN = 6
D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 4

def make_batch(B, seq_len=SEQ_LEN, vocab=VOCAB, device="cpu"):
    """Predict x[:,0] from the last position."""
    x = torch.randint(0, vocab, (B, seq_len), device=device)
    target = x[:, 0]
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
        h = self.embed(ids)
        h = self.core(h)
        # use last position
        return self.head(h[:, -1])


def compute_spectrum_averaged(model, ids_batch, n_samples=4):
    """Run lyapunov_spectrum on first n_samples examples of ids_batch and average."""
    spectra = []
    with torch.no_grad():
        embs = model.embed(ids_batch[:n_samples])    # (n, T, D)
    for i in range(n_samples):
        x0 = embs[i:i+1].detach().clone()
        lam, _ = lyapunov_spectrum(model.core, x0)
        spectra.append(lam)
    M = np.stack(spectra, axis=0)
    return M.mean(axis=0)


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    cfg = BlockConfig(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, seq_len=SEQ_LEN,
                      use_skip=True, use_ffn=True, ln_style="pre")
    model = FullModel(cfg, n_layers=N_LAYERS, vocab=VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    n_steps = 800
    eval_every = 40
    probe_batch = 4

    # fixed probe batch for repeatable spectrum measurement
    probe_ids, _ = make_batch(probe_batch)

    history = []          # list of dict
    spectra_log = []      # list of (step, lam)

    # initial measurement
    model.eval()
    lam0 = compute_spectrum_averaged(model, probe_ids, n_samples=probe_batch)
    spectra_log.append((0, lam0))
    history.append(dict(step=0, loss=float("nan"), acc=float("nan"),
                        **staircase_summary(lam0)))

    t0 = time.time()
    for step in range(1, n_steps + 1):
        model.train()
        ids, target = make_batch(64)
        logits = model(ids)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % eval_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                ids_e, tgt_e = make_batch(256)
                pred = model(ids_e).argmax(-1)
                acc = (pred == tgt_e).float().mean().item()
            lam = compute_spectrum_averaged(model, probe_ids, n_samples=probe_batch)
            spectra_log.append((step, lam))
            s = staircase_summary(lam)
            history.append(dict(step=step, loss=loss.item(), acc=acc, **s))
            print(f"step {step:4d}  loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"spread {s['spread']:.4f}  λmax {lam.max():+.4f}  λmin {lam.min():+.4f}  "
                  f"plateaus {s['n_plateaus']}")

    print(f"training done in {time.time()-t0:.1f}s")

    # --- save raw ---
    steps = np.array([s for s, _ in spectra_log])
    spec_mat = np.stack([l for _, l in spectra_log], axis=0)   # (T, n)
    np.savez(f"{OUT}/exp2_dynamics.npz",
             steps=steps, spectra=spec_mat,
             history=np.array(json.dumps(history)))

    # ====== plots ======

    # (1) heatmap of spectrum over training
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(spec_mat.T, aspect="auto", origin="lower",
                   extent=[steps[0], steps[-1], 0, spec_mat.shape[1]],
                   cmap="RdBu_r",
                   vmin=-np.abs(spec_mat).max(), vmax=np.abs(spec_mat).max())
    ax.set_xlabel("training step")
    ax.set_ylabel("rank (0=largest, top)")
    ax.set_title("Lyapunov spectrum over training (heatmap)")
    fig.colorbar(im, ax=ax, label=r"$\lambda_i$")
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2_heatmap.png", dpi=140); plt.close(fig)

    # (2) lambda_min, lambda_max, spread vs step
    lam_min = spec_mat.min(axis=1)
    lam_max = spec_mat.max(axis=1)
    spread = lam_max - lam_min

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, lam_max, label=r"$\lambda_{\max}$", color="C3")
    ax.plot(steps, lam_min, label=r"$\lambda_{\min}$", color="C0")
    ax.plot(steps, spread, label="spread", color="C2", ls="--")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("step"); ax.set_ylabel(r"$\lambda$")
    ax.set_title("Top / bottom Lyapunov exponents through training")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_top_bottom.png", dpi=140); plt.close(fig)

    # (3) top-k vs bottom-k tracks  (test the hypothesis directly)
    n = spec_mat.shape[1]
    k = max(3, n // 10)
    top_k = np.sort(spec_mat, axis=1)[:, -k:].mean(axis=1)        # avg of top k
    bot_k = np.sort(spec_mat, axis=1)[:, :k].mean(axis=1)         # avg of bottom k
    mid_k = np.sort(spec_mat, axis=1)[:, (n//2 - k//2):(n//2 + k//2)].mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, top_k, label=f"mean of top {k}",    color="C3")
    ax.plot(steps, mid_k, label=f"mean of middle {k}", color="C7")
    ax.plot(steps, bot_k, label=f"mean of bottom {k}", color="C0")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("step"); ax.set_ylabel(r"$\lambda$")
    ax.set_title("Top vs bottom band — does the bottom collapse first?")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_topbottom_bands.png", dpi=140); plt.close(fig)

    # (4) overlay of spectra at selected snapshots
    fig, ax = plt.subplots(figsize=(9, 5))
    idxs = np.linspace(0, len(steps) - 1, 6).astype(int)
    cmap = plt.get_cmap("viridis")
    for j, idx in enumerate(idxs):
        ax.plot(spec_mat[idx], color=cmap(j / max(1, len(idxs)-1)),
                label=f"step {steps[idx]}")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("rank"); ax.set_ylabel(r"$\lambda$")
    ax.set_title("Spectrum snapshots during training")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/exp2_snapshots.png", dpi=140); plt.close(fig)

    # (5) directional convergence: how much each rank moves toward 0
    # plot per-rank trajectory normalized by its initial value
    fig, ax = plt.subplots(figsize=(9, 5))
    init = spec_mat[0]                              # (n,)
    # show fraction |lambda_t| / |lambda_0| per rank, then averaged in bins
    bins = 5
    bin_ids = np.linspace(0, n, bins + 1).astype(int)
    for b in range(bins):
        s_idx = np.argsort(init)                    # ascending (most negative first)
        chunk = s_idx[bin_ids[b]:bin_ids[b + 1]]
        track = np.abs(spec_mat[:, chunk]).mean(axis=1)
        ax.plot(steps, track,
                label=f"rank-bin {b+1}/{bins} (init λ̄={init[chunk].mean():+.3f})")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\langle |\lambda| \rangle$ within bin")
    ax.set_title("Convergence rate per spectrum band (sorted by INITIAL value)")
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2_per_band_convergence.png", dpi=140)
    plt.close(fig)

    # (6) loss & accuracy
    hist_steps = [h["step"] for h in history if not np.isnan(h["loss"])]
    losses = [h["loss"] for h in history if not np.isnan(h["loss"])]
    accs = [h["acc"] for h in history if not np.isnan(h["acc"])]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(hist_steps, losses, "C3-", label="loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss", color="C3")
    ax2 = ax1.twinx()
    ax2.plot(hist_steps, accs, "C0-", label="acc")
    ax2.set_ylabel("accuracy", color="C0")
    ax1.set_title("Training curves")
    fig.tight_layout(); fig.savefig(f"{OUT}/exp2_training.png", dpi=140); plt.close(fig)

    return history, spec_mat, steps


if __name__ == "__main__":
    main()
