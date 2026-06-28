"""
Experiment 1
============
Sweep architecture parameters and study how staircase structure of the
initial (untrained) Lyapunov spectrum depends on them.

Axes:
  - skip on/off
  - FFN on/off and position (before/after attention)
  - LN style: pre / post / none
  - width d_model
  - depth n_layers

For each config we compute the spectrum at random init (averaged over a few
seeds and inputs), then quantify the staircase.

Outputs:
  - exp1_spectra.npz  (raw spectra)
  - exp1_summary.csv  (staircase metrics)
  - exp1_*.png        (plots)
"""
from __future__ import annotations
import sys, os, csv, itertools, time
sys.path.insert(0, "/home/claude")
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformer_block import BlockConfig, TinyTransformer
from lyapunov import lyapunov_spectrum
from staircase import detect_plateaus, staircase_summary

OUT = "/home/claude/out"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)

# -----------------------------------------------------------
# Sweep definitions  (kept small for runtime)
# -----------------------------------------------------------
SEQ_LEN = 6
N_HEADS = 2
N_SEEDS = 3

def run_one(cfg: BlockConfig, n_layers: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyTransformer(cfg, n_layers=n_layers)
    x0 = torch.randn(1, cfg.seq_len, cfg.d_model)
    lam, _ = lyapunov_spectrum(model, x0)
    return lam

def averaged_spectrum(cfg, n_layers, n_seeds=N_SEEDS):
    spectra = [run_one(cfg, n_layers, s) for s in range(n_seeds)]
    M = np.stack(spectra, axis=0)            # (S, n)
    return M.mean(axis=0), M.std(axis=0)

# -----------------------------------------------------------
# Sweep 1A:  ablation of skip / FFN / LN at fixed width & depth
# -----------------------------------------------------------
sweep_1a = []
for use_skip in [True, False]:
    for use_ffn in [True, False]:
        for ln_style in ["pre", "post", "none"]:
            sweep_1a.append(dict(
                use_skip=use_skip, use_ffn=use_ffn, ln_style=ln_style,
            ))

results_1a = []
print("=== Sweep 1A: ablation ===")
t0 = time.time()
for spec in sweep_1a:
    cfg = BlockConfig(d_model=12, n_heads=N_HEADS, d_ff=24, seq_len=SEQ_LEN,
                      **spec)
    mean, std = averaged_spectrum(cfg, n_layers=6)
    summ = staircase_summary(mean)
    rec = dict(spec, n_plateaus=summ["n_plateaus"],
               max_plateau_width=max(summ["plateau_widths"]) if summ["plateau_widths"] else 0,
               total_width=summ["total_width"], spread=float(mean.max() - mean.min()))
    results_1a.append((rec, mean, std))
    print(f"  skip={spec['use_skip']} ffn={spec['use_ffn']} ln={spec['ln_style']:>4s}  "
          f"plateaus={summ['n_plateaus']:2d}  max_w={rec['max_plateau_width']:2d}  "
          f"spread={rec['spread']:.4f}")
print(f"  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------
# Sweep 1B:  width sweep (d_model)
# -----------------------------------------------------------
widths = [8, 12, 16, 24, 32]
results_1b = []
print("\n=== Sweep 1B: width ===")
t0 = time.time()
for d in widths:
    cfg = BlockConfig(d_model=d, n_heads=N_HEADS, d_ff=2*d, seq_len=SEQ_LEN,
                      use_skip=True, use_ffn=True, ln_style="pre")
    mean, std = averaged_spectrum(cfg, n_layers=6)
    summ = staircase_summary(mean)
    results_1b.append(dict(d_model=d, mean=mean, std=std, summary=summ))
    print(f"  d={d:3d}  n_lambda={len(mean):3d}  plateaus={summ['n_plateaus']:2d}  "
          f"max_w={max(summ['plateau_widths']) if summ['plateau_widths'] else 0}")
print(f"  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------
# Sweep 1C:  depth sweep (n_layers)
# -----------------------------------------------------------
depths = [2, 4, 6, 8, 12, 16]
results_1c = []
print("\n=== Sweep 1C: depth ===")
t0 = time.time()
for L in depths:
    cfg = BlockConfig(d_model=12, n_heads=N_HEADS, d_ff=24, seq_len=SEQ_LEN,
                      use_skip=True, use_ffn=True, ln_style="pre")
    mean, std = averaged_spectrum(cfg, n_layers=L)
    summ = staircase_summary(mean)
    results_1c.append(dict(n_layers=L, mean=mean, std=std, summary=summ))
    print(f"  L={L:2d}  plateaus={summ['n_plateaus']:2d}  "
          f"spread={float(mean.max()-mean.min()):.4f}")
print(f"  ({time.time()-t0:.1f}s)")

# -----------------------------------------------------------
# Sweep 1D: FFN position
# -----------------------------------------------------------
results_1d = []
print("\n=== Sweep 1D: FFN position ===")
for pos in ["after", "before"]:
    cfg = BlockConfig(d_model=12, n_heads=N_HEADS, d_ff=24, seq_len=SEQ_LEN,
                      use_skip=True, use_ffn=True, ln_style="pre",
                      ffn_position=pos)
    mean, std = averaged_spectrum(cfg, n_layers=6)
    summ = staircase_summary(mean)
    results_1d.append(dict(ffn_position=pos, mean=mean, std=std, summary=summ))
    print(f"  pos={pos:6s}  plateaus={summ['n_plateaus']:2d}")

# -----------------------------------------------------------
# Save raw arrays
# -----------------------------------------------------------
np.savez(f"{OUT}/exp1_spectra.npz",
         **{f"1a_{i}": r[1] for i, r in enumerate(results_1a)},
         **{f"1b_d{r['d_model']}": r["mean"] for r in results_1b},
         **{f"1c_L{r['n_layers']}": r["mean"] for r in results_1c},
         **{f"1d_{r['ffn_position']}": r["mean"] for r in results_1d})

# Save CSV summary for 1A
with open(f"{OUT}/exp1_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["use_skip", "use_ffn", "ln_style", "n_plateaus",
                "max_plateau_width", "total_width", "spread"])
    for rec, _, _ in results_1a:
        w.writerow([rec["use_skip"], rec["use_ffn"], rec["ln_style"],
                    rec["n_plateaus"], rec["max_plateau_width"],
                    rec["total_width"], rec["spread"]])

# -----------------------------------------------------------
# Plots
# -----------------------------------------------------------
def plot_spectra(items, labels, title, fname, mark_plateaus=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, (lam, lab) in enumerate(zip(items, labels)):
        idx = np.arange(len(lam))
        ax.plot(idx, lam, marker="o", ms=3, lw=1.2, color=cmap(i % 10), label=lab)
        if mark_plateaus:
            plats, _, _ = detect_plateaus(lam)
            for p in plats:
                ax.axhline(p["value"], color=cmap(i % 10), alpha=0.08, lw=4)
    ax.set_xlabel("index (sorted descending)")
    ax.set_ylabel(r"local Lyapunov exponent $\lambda_i$")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)

# 1A: ablation grid - one figure per LN style
for ln in ["pre", "post", "none"]:
    sel = [r for r in results_1a if r[0]["ln_style"] == ln]
    items = [r[1] for r in sel]
    labels = [f"skip={r[0]['use_skip']}, ffn={r[0]['use_ffn']}" for r in sel]
    plot_spectra(items, labels,
                 f"Exp1A — Architecture ablation (LN={ln})",
                 f"{OUT}/exp1a_ablation_{ln}.png")

# 1B: width
plot_spectra([r["mean"] for r in results_1b],
             [f"d={r['d_model']}" for r in results_1b],
             "Exp1B — Width sweep (depth=6, Pre-LN, skip+FFN)",
             f"{OUT}/exp1b_width.png")

# 1B normalized by index fraction
fig, ax = plt.subplots(figsize=(8, 5))
for r in results_1b:
    lam = r["mean"]
    ax.plot(np.linspace(0, 1, len(lam)), lam, label=f"d={r['d_model']}")
ax.set_xlabel("rank / n")
ax.set_ylabel(r"$\lambda$")
ax.set_title("Exp1B — Width sweep, rescaled rank")
ax.axhline(0, color="k", lw=0.6, alpha=0.4)
ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT}/exp1b_width_rescaled.png", dpi=140); plt.close(fig)

# 1C: depth
plot_spectra([r["mean"] for r in results_1c],
             [f"L={r['n_layers']}" for r in results_1c],
             "Exp1C — Depth sweep (d_model=12, Pre-LN, skip+FFN)",
             f"{OUT}/exp1c_depth.png")

# 1C: plateau count vs depth
fig, ax = plt.subplots(figsize=(6, 4))
xs = [r["n_layers"] for r in results_1c]
ys = [r["summary"]["n_plateaus"] for r in results_1c]
ax.plot(xs, ys, "o-")
ax.set_xlabel("n_layers"); ax.set_ylabel("# plateaus")
ax.set_title("Plateau count vs depth")
ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT}/exp1c_plateau_count_vs_depth.png", dpi=140); plt.close(fig)

# 1B: plateau width vs d
fig, ax = plt.subplots(figsize=(6, 4))
xs = [r["d_model"] for r in results_1b]
ys_n = [r["summary"]["n_plateaus"] for r in results_1b]
ys_w = [max(r["summary"]["plateau_widths"]) if r["summary"]["plateau_widths"] else 0
        for r in results_1b]
ax.plot(xs, ys_n, "o-", label="# plateaus")
ax.plot(xs, ys_w, "s-", label="max plateau width")
ax.set_xlabel("d_model"); ax.set_title("Staircase metrics vs width")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/exp1b_metrics_vs_width.png", dpi=140); plt.close(fig)

# 1D: FFN position
plot_spectra([r["mean"] for r in results_1d],
             [f"ffn={r['ffn_position']}" for r in results_1d],
             "Exp1D — FFN position",
             f"{OUT}/exp1d_ffn_pos.png")

print(f"\nWrote results to {OUT}/")
