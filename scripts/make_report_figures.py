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
    nlp_entity = [0.969, 0.931, 0.795, 0.748, 0.738]
    element = [0.756, 0.577, 0.582, 0.719, 0.509]
    layout = [0.733, 0.530, 0.261, 0.688, 0.336]

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
    precision = [0.969, 0.993, 0.995, 0.940, 0.711]
    recall = [0.914, 0.802, 0.607, 0.613, 0.678]

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
    ax.set_title("Latency vs.\ quality (bubble size $\\propto$ mean response length)")
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0.65, 1.0)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_latency_quality_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os

    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_dataset_overview()
    fig2_benchmark_f1_by_type()
    fig3_mitigation_comparison()
    fig4_precision_recall_scatter()
    fig5_latency_quality_tradeoff()
    print("Figures written to", OUT_DIR)
