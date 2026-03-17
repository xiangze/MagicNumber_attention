import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict

# ──────────────────────────────────────────────
# 共通設定
# ──────────────────────────────────────────────
ZERO_EV_THRESHOLD = 0.01
TEMPLATE = "plotly_white"   # "plotly_dark" も可

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


# ──────────────────────────────────────────────
# Exp A プロット
# ──────────────────────────────────────────────
def plot_A(results: List[ExperimentResult]):
    """
    matplotlib版: 3 × plt.subplot
    Plotly版  : make_subplots(1, 3) + add_scatter/add_bar
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Hessian Eigenvalue Spectrum",
            "Flatness Ratio vs Generalization",
            "Edge of Chaos vs Flat Minimum",
        ],
    )
    # ── (1) 固有値スペクトル（折れ線）────────────────
    for r in results:
        ys = np.abs(r.eigenvalues) + 1e-10
        fig.add_scatter(
            row=1, col=1,
            x=list(range(len(ys))), y=ys,
            mode="lines+markers", name=r.label,
            marker=dict(size=5),
        )
    # τ ライン
    fig.add_hline(y=ZERO_EV_THRESHOLD, line_dash="dash",
                  line_color="gray", row=1, col=1,
                  annotation_text=f"τ={ZERO_EV_THRESHOLD}")
    fig.update_yaxes(type="log", title_text="|λ|", row=1, col=1)
    fig.update_xaxes(title_text="Index (sorted)", row=1, col=1)

    # ── (2) Flatness Ratio vs Val Acc（散布図）────────
    for r in results:
        fig.add_scatter(
            row=1, col=2,
            x=[r.flatness_ratio], y=[r.val_acc],
            mode="markers+text",
            name=r.label, text=[r.label],
            textposition="top center",
            marker=dict(size=14),
            showlegend=False,
        )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  row=1, col=2, annotation_text="chance")
    fig.update_xaxes(title_text="Flatness Ratio φ(θ)", row=1, col=2)
    fig.update_yaxes(title_text="Val Accuracy", row=1, col=2)

    # ── (3) λ vs φ ────────────────────────────────
    for r in results:
        fig.add_scatter(
            row=1, col=3,
            x=[r.lyapunov], y=[r.flatness_ratio],
            mode="markers+text",
            name=r.label, text=[r.label],
            textposition="top center",
            marker=dict(size=14),
            showlegend=False,
        )
    fig.add_vline(x=0.0, line_dash="dash", line_color="black",
                  row=1, col=3, annotation_text="λ=0")
    fig.update_xaxes(title_text="Lyapunov λ", row=1, col=3)
    fig.update_yaxes(title_text="Flatness Ratio φ", row=1, col=3)

    fig.update_layout(
        title="Exp A: Normal Training vs Random Label Training",
        template=TEMPLATE, height=450, width=1200,
    )
    fig.write_html("expA_flat_minimum.html")
    fig.write_image("expA_flat_minimum.png", scale=2)
    print("  → Saved: expA_flat_minimum.html / .png")
    return fig


# ──────────────────────────────────────────────
# Exp B プロット
# ──────────────────────────────────────────────
def plot_B(results: List[ExperimentResult]):
    """
    matplotlib版: 3 × plt.subplot + colorbar手動作成
    Plotly版  : px.scatter で color/size/hover を1行指定
    """
    df = pd.DataFrame([{
        "label":         r.label,
        "batch_size":    r.extra["batch_size"],
        "flatness_ratio": r.flatness_ratio,
        "lyapunov":      r.lyapunov,
        "val_acc":       r.val_acc,
        "train_acc":     r.train_acc,
    } for r in results])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Batch Size vs Flatness Ratio",
            "Flatness Ratio vs Generalization",
            "Lyapunov λ vs Flatness Ratio",
        ],
    )

    # ── (1) BS vs φ（折れ線）────────────────────────
    fig.add_scatter(
        row=1, col=1,
        x=df["batch_size"], y=df["flatness_ratio"],
        mode="lines+markers+text",
        text=df["label"], textposition="top center",
        marker=dict(size=10, color="steelblue"),
        name="φ",
    )
    fig.update_xaxes(type="log", title_text="Batch Size (log)", row=1, col=1)
    fig.update_yaxes(title_text="Flatness Ratio φ", row=1, col=1)

    # ── (2) φ vs Val Acc（色=log BS）──────────────
    fig.add_scatter(
        row=1, col=2,
        x=df["flatness_ratio"], y=df["val_acc"],
        mode="markers+text",
        text=df["label"], textposition="top center",
        marker=dict(
            size=14,
            color=np.log10(df["batch_size"]),
            colorscale="RdYlGn_r",         # 大BS=赤=sharp
            colorbar=dict(title="log10(BS)", x=0.63),
            showscale=True,
        ),
        name="BS scan",
        customdata=df[["batch_size", "val_acc"]].values,
        hovertemplate="BS=%{customdata[0]}<br>val_acc=%{customdata[1]:.3f}",
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Flatness Ratio φ", row=1, col=2)
    fig.update_yaxes(title_text="Val Accuracy",     row=1, col=2)

    # ── (3) λ vs φ ────────────────────────────────
    fig.add_scatter(
        row=1, col=3,
        x=df["lyapunov"], y=df["flatness_ratio"],
        mode="markers+text",
        text=df["label"], textposition="top center",
        marker=dict(
            size=14,
            color=np.log10(df["batch_size"]),
            colorscale="RdYlGn_r",
        ),
        showlegend=False,
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="black",
                  row=1, col=3, annotation_text="λ=0")
    fig.update_xaxes(title_text="Lyapunov λ", row=1, col=3)
    fig.update_yaxes(title_text="Flatness Ratio φ", row=1, col=3)

    fig.update_layout(
        title="Exp B: Batch Size → Sharp vs Flat Minimum",
        template=TEMPLATE, height=450, width=1200,
    )
    fig.write_html("expB_batch_flatness.html")
    fig.write_image("expB_batch_flatness.png", scale=2)
    print("  → Saved: expB_batch_flatness.html / .png")
    return fig


# ──────────────────────────────────────────────
# Exp C プロット
# ──────────────────────────────────────────────
def plot_C(results_normal: List[ExperimentResult],
           results_random: List[ExperimentResult]):
    """
    matplotlib版: gridspec + ax.scatter3d 手動
    Plotly版  :
      - 2D散布図 → px.scatter (1行)
      - 3D散布図 → px.scatter_3d (1行)
      - 棒グラフ  → px.bar (1行)
      - subplot合成は make_subplots
    """
    all_results = results_normal + results_random

    df = pd.DataFrame([{
        "label":          r.label,
        "training_type":  r.extra.get("training_type", "?"),
        "lr":             r.extra.get("lr", np.nan),
        "log_lr":         np.log10(r.extra["lr"]) if r.extra.get("lr") else np.nan,
        "flatness_ratio": r.flatness_ratio,
        "lyapunov":       r.lyapunov,
        "val_acc":        r.val_acc,
        "train_acc":      r.train_acc,
    } for r in all_results])

    # ── (1) φ vs Val Acc ──────────────────────────
    fig1 = px.scatter(
        df, x="flatness_ratio", y="val_acc",
        color="training_type", symbol="training_type",
        hover_data=["label", "lr", "lyapunov"],
        title="Flatness Ratio vs Generalization",
        labels={"flatness_ratio": "φ(θ)", "val_acc": "Val Accuracy"},
        template=TEMPLATE,
    )
    fig1.add_hline(y=0.5, line_dash="dot", line_color="gray",
                   annotation_text="chance level")

    # ── (2) λ vs Val Acc ──────────────────────────
    fig2 = px.scatter(
        df, x="lyapunov", y="val_acc",
        color="training_type", symbol="training_type",
        hover_data=["label", "lr", "flatness_ratio"],
        title="Lyapunov λ vs Generalization<br>(λ≈0 with low acc = counterexample)",
        labels={"lyapunov": "Lyapunov λ", "val_acc": "Val Accuracy"},
        template=TEMPLATE,
    )
    fig2.add_vline(x=0.0, line_dash="dash", line_color="black",  annotation_text="λ=0 (Edge of Chaos)")
    fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")

    # ── (3) λ × φ 散布図（色=Val Acc）──────────────
    fig3 = px.scatter(
        df, x="lyapunov", y="flatness_ratio",
        color="val_acc", symbol="training_type",
        color_continuous_scale="RdYlGn",
        range_color=[0.4, 1.0],
        hover_data=["label", "lr"],
        title="λ × φ Joint Space<br>(Color = Val Accuracy)",
        labels={"lyapunov": "Lyapunov λ", "flatness_ratio": "φ(θ)"},
        template=TEMPLATE,
    )
    fig3.add_vline(x=0.0, line_dash="dash", line_color="black")

    # ── (4) 3D: log_lr × φ × λ（色=Val Acc）────────
    fig4 = px.scatter_3d(
        df, x="log_lr", y="flatness_ratio", z="lyapunov",
        color="val_acc", symbol="training_type",
        color_continuous_scale="RdYlGn",
        range_color=[0.4, 1.0],
        hover_data=["label"],
        title="3D: log(LR) × φ × λ  (Color = Generalization)",
        labels={
            "log_lr": "log10(LR)",
            "flatness_ratio": "φ(θ)",
            "lyapunov": "Lyapunov λ",
        },
        template=TEMPLATE,
    )

    # ── (5) 予測力比較（棒グラフ）──────────────────
    phi  = df["flatness_ratio"].values
    lya  = df["lyapunov"].values
    acc  = df["val_acc"].values
    corr_phi  = abs(np.corrcoef(phi,       acc)[0, 1])
    corr_lya  = abs(np.corrcoef(lya,       acc)[0, 1])
    corr_comb = abs(np.corrcoef(phi - lya, acc)[0, 1])

    df_bar = pd.DataFrame({
        "Metric": ["φ alone", "λ alone", "φ−λ combined"],
        "|Pearson r|": [corr_phi, corr_lya, corr_comb],
        "Color": ["steelblue", "salmon", "mediumpurple"],
    })
    fig5 = px.bar(
        df_bar, x="Metric", y="|Pearson r|",
        color="Metric",
        color_discrete_map={
            "φ alone": "steelblue",
            "λ alone": "salmon",
            "φ−λ combined": "mediumpurple",
        },
        text=df_bar["|Pearson r|"].map("{:.3f}".format),
        title="Predictive Power: |Pearson r| with Val Accuracy",
        range_y=[0, 1],
        template=TEMPLATE,
    )
    fig5.update_traces(textposition="outside")

    # 全図を HTML 1ファイルに保存
    _save_multi_html(
        [fig1, fig2, fig3, fig4, fig5],
        titles=["φ vs Acc", "λ vs Acc", "λ×φ space", "3D map", "Predictive power"],
        filename="expC_3d_map.html",
    )
    # PNG はサブプロット結合版
    fig4.write_image("expC_3d_map.png", scale=2)
    print("  → Saved: expC_3d_map.html / expC_3d_map.png")
    return fig1, fig2, fig3, fig4, fig5


def _save_multi_html(figs, titles, filename):
    """複数のPlotly figをタブ切り替えで1 HTMLにまとめる"""
    tabs_html = ""
    plots_html = ""
    for i, (fig, title) in enumerate(zip(figs, titles)):
        active = "active" if i == 0 else ""
        tabs_html  += f'<button class="tab {active}" onclick="show({i})">{title}</button>\n'
        display     = "block" if i == 0 else "none"
        plots_html += (f'<div id="plot{i}" style="display:{display}">'
                       + fig.to_html(full_html=False, include_plotlyjs=(i == 0))
                       + "</div>\n")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body {{ font-family: sans-serif; }}
  .tab {{ padding: 8px 16px; margin: 2px; cursor: pointer;
          border: 1px solid #ccc; background: #f0f0f0; border-radius: 4px; }}
  .tab.active {{ background: #4e79a7; color: white; }}
</style></head><body>
<div style="padding:12px">
  <h3>Exp C: 3D Classification Map of Generalization Metrics</h3>
  <div>{tabs_html}</div>
  {plots_html}
</div>
<script>
function show(i) {{
  document.querySelectorAll('[id^=plot]').forEach((el,j) => {{
    el.style.display = j===i ? 'block' : 'none';
  }});
  document.querySelectorAll('.tab').forEach((el,j) => {{
    el.classList.toggle('active', j===i);
  }});
}}
</script></body></html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


# ──────────────────────────────────────────────
# 閾値感度分析
# ──────────────────────────────────────────────
def analyze_threshold_sensitivity(result: ExperimentResult,
                                   tau_range=None):
    """
    matplotlib版: plt.semilogx 1本
    Plotly版  : px.line (1行) + log_x=True
    """
    if tau_range is None:
        tau_range = np.logspace(-4, 1, 60)

    ratios = [
        np.sum(np.abs(result.eigenvalues) < tau) / len(result.eigenvalues)
        for tau in tau_range
    ]

    df = pd.DataFrame({"tau": tau_range, "flatness_ratio": ratios})
    fig = px.line(
        df, x="tau", y="flatness_ratio",
        log_x=True,
        markers=True,
        title=f"Threshold Sensitivity: '{result.label}'<br>"
              "φ(θ) depends on τ — intrinsic limitation",
        labels={"tau": "Threshold τ", "flatness_ratio": "Flatness Ratio φ(θ)"},
        template=TEMPLATE,
    )
    fig.add_vline(
        x=ZERO_EV_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text=f"default τ={ZERO_EV_THRESHOLD}",
    )
    fig.write_html("bonus_threshold_sensitivity.html")
    fig.write_image("bonus_threshold_sensitivity.png", scale=2)
    print("  → Saved: bonus_threshold_sensitivity.html / .png")
    return fig


# ──────────────────────────────────────────────
# seaborn補助: 固有値ヒートマップ（複数モデル比較）
# ──────────────────────────────────────────────
def plot_eigenvalue_heatmap(results: List[ExperimentResult],
                             top_k: int = 30):
    """
    複数モデルの固有値スペクトルを seaborn heatmap で比較。
    Plotlyには heatmap もあるが、seabornの方が1行で美しい。
    """
    max_k = min(top_k, min(len(r.eigenvalues) for r in results))
    data = np.vstack([
        np.abs(r.eigenvalues[:max_k]) for r in results
    ])
    df_heat = pd.DataFrame(
        data,
        index=[r.label for r in results],
        columns=[f"λ{i+1}" for i in range(max_k)],
    )

    fig, ax = plt.subplots(figsize=(min(max_k * 0.5 + 2, 16), len(results) * 1.2 + 1))
    sns.heatmap(
        np.log10(df_heat + 1e-10),
        ax=ax, cmap="RdYlGn_r",
        cbar_kws={"label": "log10|λ|"},
        linewidths=0.3, annot=(max_k <= 15),
        fmt=".1f" if max_k <= 15 else "",
    )
    ax.set_title(f"Eigenvalue Heatmap (top-{max_k} | log scale)\n"
                 "Dark = near-zero (flat direction)")
    plt.tight_layout()
    plt.savefig("eigenvalue_heatmap.png", dpi=130, bbox_inches="tight")
    print("  → Saved: eigenvalue_heatmap.png")
    plt.close()

# ──────────────────────────────────────────────
# 動作確認用ダミーデータ
# ──────────────────────────────────────────────
def make_dummy_results():
    rng = np.random.default_rng(42)
    results = []
    configs = [
        ("Normal_lr1e-5", True,  1e-5, 16),
        ("Normal_lr1e-4", True,  1e-4, 16),
        ("Normal_lr1e-3", True,  1e-3, 16),
        ("Random_lr1e-5", False, 1e-5, 16),
        ("Random_lr1e-4", False, 1e-4, 16),
        ("BS4_normal",    True,  2e-5,  4),
        ("BS64_normal",   True,  2e-5, 64),
        ("BS256_normal",  True,  2e-5, 256),
    ]
    for label, is_normal, lr, bs in configs:
        n_ev = 20
        if is_normal:
            # flat: 多くの固有値がゼロ付近
            eigs = np.abs(rng.normal(0, 0.005 * (1 + np.log10(bs)), n_ev))
            eigs[:3] = rng.uniform(0.1, 2.0, 3)  # 少数の大固有値
            flat_ratio = (eigs < ZERO_EV_THRESHOLD).mean()
            val_acc = np.clip(rng.normal(0.82 - 0.1 * np.log10(bs), 0.03), 0.5, 0.98)
            lya = rng.normal(-0.05, 0.03)
        else:
            # random: 全体的にsharp
            eigs = np.abs(rng.exponential(0.5, n_ev))
            flat_ratio = (eigs < ZERO_EV_THRESHOLD).mean()
            val_acc = np.clip(rng.normal(0.51, 0.02), 0.48, 0.58)
            lya = rng.normal(-0.15, 0.05)

        results.append(ExperimentResult(
            label=label,
            eigenvalues=np.sort(eigs)[::-1],
            flatness_ratio=flat_ratio,
            effective_rank_ev=float(np.exp(-np.sum(
                (eigs / eigs.sum()) * np.log(eigs / eigs.sum() + 1e-10)))),
            lyapunov=float(lya),
            val_acc=float(val_acc),
            train_acc=float(val_acc + rng.uniform(0.02, 0.12)),
            extra={"training_type": "Normal" if is_normal else "Random",
                   "lr": lr, "batch_size": bs},
        ))
    return results


if __name__ == "__main__":
    print("Plotly visualization demo (dummy data)\n")

    all_res = make_dummy_results()
    normal  = [r for r in all_res if r.extra["training_type"] == "Normal"]
    random_ = [r for r in all_res if r.extra["training_type"] == "Random"]
    ab_res  = [r for r in all_res if "BS" not in r.label][:2]  # A用

    print("[Exp A plots]")
    plot_A(ab_res[:2] if len(ab_res) >= 2 else all_res[:2])

    print("[Exp B plots]")
    bs_res = [r for r in all_res if "BS" in r.label]
    if bs_res:
        plot_B(bs_res)

    print("[Exp C plots]")
    plot_C(normal, random_)

    print("[Threshold sensitivity]")
    analyze_threshold_sensitivity(all_res[0])

    print("[Eigenvalue heatmap]")
    plot_eigenvalue_heatmap(all_res[:5])

    print("\nDone. Open .html files in a browser for interactive plots.")
