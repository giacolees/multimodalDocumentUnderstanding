"""Generate the report figures from results/ data. Run with: uv run python scripts/make_report_figures.py"""

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figs"

plt.rcParams.update(
    {
        "font.size": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)

MODEL_COLORS = {
    "Qwen2.5-VL-7B": "#4C72B0",
    "Qwen3-VL-4B": "#DD8452",
    "Qwen3.5-2B": "#55A868",
    "Gemma-4-E2B": "#C44E52",
    "Gemma-4-E2B (no-think)": "#937860",
    "SigLIP": "#8172B2",
}
TYPE_COLORS = {"nlp_entity": "#4C72B0", "element": "#DD8452", "layout": "#55A868"}


def _bar_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, fmt.format(h), ha="center", va="bottom", fontsize=6.5)


def fig1_dataset_overview():
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    types = ["nlp_entity", "element", "layout"]
    shares = [84.5, 9.7, 5.8]
    colors = [TYPE_COLORS[t] for t in types]
    bars = axes[0].bar(types, shares, color=colors)
    axes[0].set_ylabel("% of dataset")
    axes[0].set_title("Corruption-type distribution")
    _bar_labels(axes[0], bars, fmt="{:.1f}%")
    axes[0].set_ylim(0, 100)

    sources = ["DocVQA", "MP-DocVQA"]
    counts = [1758, 2560]
    bars = axes[1].bar(sources, counts, color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("# samples")
    axes[1].set_title("Source dataset composition\n(4,318 total, balanced 2,159/2,159)")
    for b, v in zip(bars, counts):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 40, str(v), ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_dataset_overview.png", bbox_inches="tight")
    plt.close(fig)


def fig2_benchmark_f1_by_type():
    models = ["Qwen2.5-VL-7B", "Qwen3-VL-4B", "Qwen3.5-2B", "Gemma-4-E2B", "SigLIP"]
    nlp_entity = [0.969, 0.931, 0.795, 0.748, 0.865]
    element = [0.756, 0.577, 0.582, 0.719, 0.693]
    layout = [0.733, 0.530, 0.261, 0.688, 0.318]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 3.3))
    b1 = ax.bar(x - width, nlp_entity, width, label="nlp_entity", color=TYPE_COLORS["nlp_entity"])
    b2 = ax.bar(x, element, width, label="element", color=TYPE_COLORS["element"])
    b3 = ax.bar(x + width, layout, width, label="layout", color=TYPE_COLORS["layout"])
    for b in (b1, b2, b3):
        _bar_labels(ax, b)

    ax.set_ylabel("F1 score")
    ax.set_title("Per-corruption-type F1 across benchmarked models (test split, n=2,158)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", zorder=0)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_benchmark_f1_by_type.png", bbox_inches="tight")
    plt.close(fig)


def fig3_mitigation_comparison():
    strategies = ["Baseline", "Few-shot", "RAG-dense", "RAG-hybrid", "RAG-BM25", "No-think\n(gemma only)"]
    nlp_entity = [0.795, 0.893, 0.825, 0.820, 0.814, 0.864]
    element = [0.582, 0.658, 0.528, 0.507, 0.493, 0.673]
    layout = [0.261, 0.541, 0.123, 0.152, 0.123, 0.286]

    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    b1 = ax.bar(x - width, nlp_entity, width, label="nlp_entity", color=TYPE_COLORS["nlp_entity"])
    b2 = ax.bar(x, element, width, label="element", color=TYPE_COLORS["element"])
    b3 = ax.bar(x + width, layout, width, label="layout", color=TYPE_COLORS["layout"])
    for b in (b1, b2, b3):
        _bar_labels(ax, b)

    ax.set_ylabel("F1 score")
    ax.set_title("Mitigation strategies vs. Qwen3.5-2B baseline (per corruption type)")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_mitigation_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig4_precision_recall_scatter():
    """Diagnoses the central Part 2 finding: precision is uniformly high,
    recall is the discriminating axis between strong and weak models."""
    models = ["Qwen2.5-VL-7B", "Qwen3-VL-4B", "Qwen3.5-2B", "Gemma-4-E2B", "SigLIP"]
    precision = [0.969, 0.993, 0.995, 0.940, 0.828]
    recall = [0.914, 0.802, 0.607, 0.613, 0.815]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m, p, r in zip(models, precision, recall):
        ax.scatter(r, p, s=130, color=MODEL_COLORS[m], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(m, (r, p), textcoords="offset points", xytext=(7, 5), fontsize=8)

    lims = (0.55, 1.03)
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, zorder=1, label="precision = recall")
    ax.fill_between(lims, lims, [lims[1], lims[1]], color="gray", alpha=0.06, zorder=0)
    ax.text(0.57, 0.985, "high precision,\nlow recall zone", fontsize=7.5, color="gray", style="italic")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs. recall, overall (test split, $n=2{,}158$)")
    ax.legend(loc="lower right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_precision_recall_scatter.png", bbox_inches="tight")
    plt.close(fig)


def fig5_latency_quality_tradeoff():
    """Bubble chart: inference latency vs. F1, bubble size = mean response
    length. Makes the no-think speed/quality tradeoff visually explicit
    against the four generative baselines."""
    points = [
        ("Qwen2.5-VL-7B", 1.43, 0.940, 38, MODEL_COLORS["Qwen2.5-VL-7B"]),
        ("Qwen3-VL-4B", 1.19, 0.887, 17, MODEL_COLORS["Qwen3-VL-4B"]),
        ("Qwen3.5-2B", 1.26, 0.754, 197, MODEL_COLORS["Qwen3.5-2B"]),
        ("Gemma-4-E2B\n(reasoning)", 1.99, 0.742, 707, MODEL_COLORS["Gemma-4-E2B"]),
        ("Gemma-4-E2B\n(no-think)", 0.46, 0.828, 43, MODEL_COLORS["Gemma-4-E2B (no-think)"]),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, lat, f1, resp_len, color in points:
        size = 80 + resp_len * 1.1
        ax.scatter(lat, f1, s=size, color=color, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(name, (lat, f1), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=7.5)

    gemma_reason = points[3]
    gemma_nothink = points[4]
    ax.annotate(
        "",
        xy=(gemma_nothink[1], gemma_nothink[2]),
        xytext=(gemma_reason[1], gemma_reason[2]),
        arrowprops=dict(arrowstyle="->", color="#937860", lw=1.2, linestyle="dotted"),
    )

    ax.set_xlabel("Mean inference latency (s)")
    ax.set_ylabel("Overall F1")
    ax.set_title("Latency vs. quality (bubble size $\\propto$ mean response length)")
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0.65, 1.0)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_latency_quality_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig6_siglip_architecture():
    """Block diagram of the SigLIP classifier baseline: frozen image/text encoders
    feeding a trained bidirectional cross-attention fusion head (replaces the
    earlier plain-concat MLP head). Static schematic, no data."""
    fig, ax = plt.subplots(figsize=(6.8, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.0, 9.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=7.5, frozen=False):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)
        if frozen:
            ax.text(x + w - 0.08, y + h - 0.05, "❄", ha="right", va="top", fontsize=9, color="#4C72B0", zorder=3)
        return (x, y, w, h)

    def center(b, side="bottom"):
        x, y, w, h = b
        return {
            "top": (x + w / 2, y + h),
            "bottom": (x + w / 2, y),
            "left": (x, y + h / 2),
            "right": (x + w, y + h / 2),
        }[side]

    def arrow(p0, p1, color="black", style="-", label=None, lw=1.1):
        ax.annotate(
            "", xy=p1, xytext=p0,
            arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=style),
            zorder=1,
        )
        if label:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx, my, label, fontsize=6, color=color, ha="center", va="center",
                    backgroundcolor="white", zorder=4)

    doc = box(0.3, 8.3, 2.2, 0.7, "Document image", "#EFEFEF")
    q = box(7.5, 8.3, 2.2, 0.7, "Question text", "#EFEFEF")

    img_enc = box(0.0, 6.9, 2.8, 1.0, "SigLIP image encoder\n(so400m-patch14-384)", "#8172B2", frozen=True)
    txt_enc = box(7.2, 6.9, 2.8, 1.0, "MiniLM text encoder\n(all-MiniLM-L6-v2)", "#55A868", frozen=True)

    img_vec = box(0.3, 5.7, 2.2, 0.7, "image embed (1152-d)", "#D8D2EC")
    txt_vec = box(7.5, 5.7, 2.2, 0.7, "text embed (384-d)", "#D7E6D6")

    img_proj = box(0.3, 4.5, 2.2, 0.7, "Linear proj → 256-d", "#B9AEDC")
    txt_proj = box(7.5, 4.5, 2.2, 0.7, "Linear proj → 256-d", "#AED3AC")

    t2i = box(1.1, 2.9, 3.3, 1.0, "Text$\\to$Image\ncross-attention (4 heads)", "#C44E52", fontsize=7.5)
    i2t = box(5.6, 2.9, 3.3, 1.0, "Image$\\to$Text\ncross-attention (4 heads)", "#C44E52", fontsize=7.5)

    text_fused = box(1.1, 1.7, 2.2, 0.7, "text fused (256-d)", "#F0C9CC")
    image_fused = box(6.7, 1.7, 2.2, 0.7, "image fused (256-d)", "#F0C9CC")

    concat = box(3.4, 0.55, 3.2, 0.65, "concat (512-d)", "#FFFFFF")
    head = box(3.1, -0.85, 3.8, 1.0, "Trained MLP head\n512→256→128→1\n+ ReLU, sigmoid", "#C44E52", fontsize=7.5)
    out = box(3.7, -1.9, 2.6, 0.6, "p(unanswerable)", "#EFEFEF")

    arrow(center(doc, "bottom"), center(img_enc, "top"))
    arrow(center(q, "bottom"), center(txt_enc, "top"))
    arrow(center(img_enc, "bottom"), center(img_vec, "top"))
    arrow(center(txt_enc, "bottom"), center(txt_vec, "top"))
    arrow(center(img_vec, "bottom"), center(img_proj, "top"))
    arrow(center(txt_vec, "bottom"), center(txt_proj, "top"))

    arrow(center(txt_proj, "bottom"), (t2i[0] + 0.6, t2i[1] + t2i[3]), label="Q")
    arrow(center(img_proj, "bottom"), (t2i[0] + 2.6, t2i[1] + t2i[3]), color="#4C72B0", label="K,V")
    arrow(center(img_proj, "bottom"), (i2t[0] + 0.6, i2t[1] + i2t[3]), color="#4C72B0", label="Q")
    arrow(center(txt_proj, "bottom"), (i2t[0] + 2.6, i2t[1] + i2t[3]), label="K,V")

    arrow(center(t2i, "bottom"), center(text_fused, "top"), label="+residual, LN")
    arrow(center(i2t, "bottom"), center(image_fused, "top"), label="+residual, LN")

    arrow(center(text_fused, "bottom"), (concat[0] + 0.8, concat[1] + concat[3]))
    arrow(center(image_fused, "bottom"), (concat[0] + concat[2] - 0.8, concat[1] + concat[3]))
    arrow(center(concat, "bottom"), center(head, "top"))
    arrow(center(head, "bottom"), center(out, "top"))

    ax.text(0.0, 9.05, "❄ = frozen, pretrained", fontsize=7.5, color="#4C72B0")
    ax.text(6.3, 9.05, "red = trainable", fontsize=7.5, color="#C44E52")
    ax.set_title(
        "SigLIP classifier baseline: frozen dual encoders + bidirectional\ncross-attention fusion head",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/06_siglip_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig7_mitigation_heatmap():
    """Two-panel heatmap: delta vs. each model's own baseline, for overall F1
    (left, shows no strategy generalises) and layout F1 (right, shows RAG
    regresses layout for every model without exception)."""
    models = ["Qwen2.5-VL-7B", "Qwen3-VL-4B", "Qwen3.5-2B", "Gemma-4-E2B"]
    strategies = ["Few-shot", "RAG-dense", "RAG-hybrid", "RAG-BM25"]

    overall_delta = np.array(
        [
            [-0.027, -0.047, -0.041, -0.043],
            [0.055, 0.040, 0.039, 0.042],
            [0.103, 0.021, 0.015, 0.008],
            [0.024, 0.072, 0.067, 0.076],
        ]
    )
    layout_delta = np.array(
        [
            [0.073, -0.639, -0.610, -0.641],
            [0.137, -0.157, -0.179, -0.135],
            [0.280, -0.138, -0.109, -0.138],
            [-0.625, -0.657, -0.567, -0.570],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    titles = ["$\\Delta$ overall F1 vs. baseline", "$\\Delta$ layout F1 vs. baseline"]
    for ax, data, title in zip(axes, (overall_delta, layout_delta), titles):
        vmax = np.abs(data).max()
        im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7, color=color)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mitigation effect by model and strategy (vs. each model's own baseline)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/07_mitigation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_dataset_overview()
    fig2_benchmark_f1_by_type()
    fig3_mitigation_comparison()
    fig4_precision_recall_scatter()
    fig5_latency_quality_tradeoff()
    fig6_siglip_architecture()
    fig7_mitigation_heatmap()
    print("Figures written to", OUT_DIR)
