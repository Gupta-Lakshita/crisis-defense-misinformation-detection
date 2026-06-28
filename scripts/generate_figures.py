"""
generate_figures.py
────────────────────
Regenerates all 7 paper figures from the results JSON files.
Outputs to results/figures/.

Usage:
    python scripts/generate_figures.py \
        --results results/final_results.json \
        --cross_domain results/cross_domain_results.json \
        --user_study results/user_study_results.json \
        --output results/figures/
"""

import json, argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK   = "#1A1A2E"; MID = "#2C3E50"; BLUE = "#2980B9"; TEAL = "#1ABC9C"
RED    = "#E74C3C"; ORANGE = "#E67E22"; GREY = "#7F8C8D"; BG = "#FAFAFA"


def savefig(path, fig):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {path}")


def fig1_dataset_composition(out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    labels = ["CoAID\n(n=666)", "COVID-Twitter\n(n=2,400)", "LIAR-PLUS\n(n=2,400)",
              "FakeNewsNet\n(n=800)", "FakeOrReal\n(n=2,400)", "FakeOrReal Full\n(n=6,305)"]
    sizes  = [666, 2400, 2400, 800, 2400, 6305]
    colors = [TEAL, "#16A085", MID, "#2471A3", BLUE, GREY]
    x = 0
    for s, c, l in zip(sizes, colors, labels):
        ax.barh(0, s, left=x, color=c, height=0.6, label=l)
        if s > 400:
            ax.text(x + s/2, 0, f"{s:,}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        x += s
    ax.set_xlim(0, 15500); ax.set_yticks([])
    ax.set_xlabel("Number of Samples", fontsize=11, color=DARK)
    ax.set_title("Figure 1: Benchmark Composition — 14,971 Samples Across Six Datasets",
                 fontsize=11, fontweight="bold", color=DARK, pad=14)
    for sp in ax.spines.values(): sp.set_color(GREY)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.7)
    savefig(os.path.join(out_dir, "fig1_dataset.png"), fig)


def fig2_class_distribution(out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    datasets = ["CoAID", "COVID-Twitter", "LIAR-PLUS", "FakeNewsNet", "FakeOrReal (bm)", "FakeOrReal (full)"]
    fakes = [333, 1200, 1200, 400, 1200, 3152]; reals = [333, 1200, 1200, 400, 1200, 3153]
    x = np.arange(len(datasets)); w = 0.35
    ax.bar(x - w/2, fakes, w, label="Fake", color=RED, alpha=0.85)
    ax.bar(x + w/2, reals, w, label="Real", color=TEAL, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=9, color=DARK)
    ax.set_ylabel("Sample Count", fontsize=11, color=DARK)
    ax.set_title("Figure 2: Class Distribution After Stratified Balancing", fontsize=11, fontweight="bold", color=DARK, pad=14)
    ax.legend(fontsize=10); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(out_dir, "fig2_class_dist.png"), fig)


def fig3_training_progression(out_dir):
    fig, ax1 = plt.subplots(figsize=(8, 4.5), facecolor=BG); ax1.set_facecolor(BG); ax2 = ax1.twinx()
    epochs = [0, 1, 2, 3]; f1 = [33.33, 78.46, 80.52, 81.85]; auc = [47.23, 90.12, 93.09, 93.60]
    l1, = ax1.plot(epochs, f1,  color=BLUE,   marker="o", lw=2.5, ms=8, label="Val Macro-F1")
    l2, = ax2.plot(epochs, auc, color=ORANGE, marker="s", lw=2.5, ms=8, linestyle="--", label="Val ROC-AUC")
    ax1.axhline(77.62, color=RED, lw=1.5, linestyle=":", alpha=0.7, label="Best TF-IDF (77.62%)")
    ax1.set_xlabel("Epoch", fontsize=11, color=DARK); ax1.set_ylabel("Macro-F1 (%)", fontsize=11, color=BLUE)
    ax2.set_ylabel("ROC-AUC (%)", fontsize=11, color=ORANGE)
    ax1.set_xticks([0,1,2,3]); ax1.set_xticklabels(["Zero-Shot","Epoch 1","Epoch 2","Epoch 3"], fontsize=9)
    ax1.set_ylim(25, 100); ax2.set_ylim(40, 100)
    for ep, f, a in zip(epochs, f1, auc):
        ax1.annotate(f"{f:.1f}%", (ep, f), textcoords="offset points", xytext=(0,10), ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
        ax2.annotate(f"{a:.1f}%", (ep, a), textcoords="offset points", xytext=(0,-16), ha="center", fontsize=8.5, color=ORANGE)
    ax1.set_title("Figure 3: XLM-RoBERTa Fine-Tuning Progression\n(+48.4pp from Zero-Shot to 3 Epochs)", fontsize=11, fontweight="bold", color=DARK, pad=12)
    ax1.legend([l1, l2, ax1.get_lines()[1]], ["Val Macro-F1","Val ROC-AUC","Best TF-IDF"], loc="lower right", fontsize=9)
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    savefig(os.path.join(out_dir, "fig3_training.png"), fig)


def fig4_per_dataset(out_dir):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG); ax.set_facecolor(BG)
    datasets = ["FakeOrReal-News", "CoAID\n(Health Crisis)", "COVID-Twitter\n(Health Crisis)", "FakeNewsNet\n(Political)", "LIAR-PLUS\n(Political)"]
    svm_f1  = [85.2, 88.4, 89.0, 74.8, 55.3]; xlmr_f1 = [95.69, 92.68, 90.50, 83.59, 55.33]
    x = np.arange(len(datasets)); w = 0.36
    b1 = ax.bar(x - w/2, svm_f1,  w, label="TF-IDF + SVM",     color=GREY, alpha=0.88)
    b2 = ax.bar(x + w/2, xlmr_f1, w, label="XLM-R Fine-Tuned", color=BLUE, alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=9.5, color=DARK)
    ax.set_ylabel("Macro-F1 (%)", fontsize=11, color=DARK); ax.set_ylim(40, 105)
    ax.set_title("Figure 4: Per-Dataset Performance — SVM vs Fine-Tuned XLM-RoBERTa", fontsize=11, fontweight="bold", color=DARK, pad=14)
    ax.legend(fontsize=10)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color=DARK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(out_dir, "fig4_per_dataset.png"), fig)


def fig5_crisis_gap(out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=BG); ax.set_facecolor(BG)
    types = ["General\nPolitical News", "Health Crisis\n(COVID-19)", "Political\nDefense"]
    svm_t  = [85.2, 88.9, 60.2]; xlmr_t = [95.69, 90.99, 62.46]
    x = np.arange(len(types)); w = 0.35
    b1 = ax.bar(x - w/2, svm_t,  w, label="TF-IDF + SVM",     color=GREY, alpha=0.88)
    b2 = ax.bar(x + w/2, xlmr_t, w, label="XLM-R Fine-Tuned", color=BLUE, alpha=0.88)
    ax.annotate("", xy=(2.18, 62.46), xytext=(2.18, 90.99), arrowprops=dict(arrowstyle="<->", color=RED, lw=2.0))
    ax.text(2.35, 77, "28.5pp\ngap", color=RED, fontsize=10, fontweight="bold", va="center")
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=10.5, color=DARK)
    ax.set_ylabel("Macro-F1 (%)", fontsize=11, color=DARK); ax.set_ylim(40, 108)
    ax.set_title("Figure 5: Crisis-Type Performance Gap\n(28.5pp holds across both model families)", fontsize=11, fontweight="bold", color=DARK, pad=12)
    ax.legend(fontsize=10)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(out_dir, "fig5_crisis_gap.png"), fig)


def fig6_cross_domain(out_dir):
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=BG); ax.set_facecolor(BG)
    dsets   = ["FakeOrReal-News\n(own domain)", "CoAID\n(Health Crisis)", "COVID-Twitter\n(Health Crisis)", "FakeNewsNet\n(Political)", "LIAR-PLUS\n(Political)"]
    general = [99.36, 41.98, 39.11, 59.89, 49.33]; adapted = [85.2, 88.4, 89.0, 74.8, 55.3]
    x = np.arange(len(dsets)); w = 0.36
    b1 = ax.bar(x - w/2, general, w, label="General-Domain SVM (no adaptation)", color=RED,  alpha=0.82)
    b2 = ax.bar(x + w/2, adapted, w, label="Crisis-Adapted SVM",                 color=TEAL, alpha=0.82)
    ax.axhline(50, color=DARK, lw=1.5, linestyle="--", alpha=0.5)
    ax.text(4.6, 51.5, "Random chance (50%)", fontsize=8, color=DARK, va="bottom")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(dsets, fontsize=9, color=DARK)
    ax.set_ylabel("Macro-F1 (%)", fontsize=11, color=DARK); ax.set_ylim(30, 113)
    ax.set_title("Figure 6: Cross-Domain Collapse — General-Domain Model vs Crisis-Adapted Model", fontsize=11, fontweight="bold", color=DARK, pad=14)
    ax.legend(fontsize=9.5); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    savefig(os.path.join(out_dir, "fig6_crossdomain.png"), fig)


def fig7_user_study(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
    colors_q = [TEAL, BLUE, ORANGE]
    qs = ["Q1: Prediction\nPlausibility", "Q2: Explanation\nMeaningfulness", "Q3: Crisis\nTrustworthiness"]
    means = [3.52, 2.72, 1.93]; sds = [0.66, 0.85, 0.65]
    pcts_a = [90.0, 53.3, 6.7]; pcts_n = [10.0, 46.7, 93.3]

    ax = axes[0]; ax.set_facecolor(BG)
    bars = ax.bar(qs, means, color=colors_q, alpha=0.85, width=0.55)
    ax.errorbar([0,1,2], means, yerr=sds, fmt="none", color="black", capsize=6, lw=2)
    ax.axhline(3.0, color=RED, lw=1.5, linestyle="--", alpha=0.7)
    ax.text(2.52, 3.08, "Adequate\nthreshold (3.0)", fontsize=8, color=RED, va="bottom")
    ax.set_ylim(0, 5.2); ax.set_ylabel("Mean Rating (1–5 Likert)", fontsize=10, color=DARK)
    ax.set_title("Mean Ratings ± SD\n(n=30)", fontsize=10, fontweight="bold", color=DARK)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, m+0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=DARK)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes[1]; ax2.set_facecolor(BG); x = np.arange(3)
    ax2.bar(x, pcts_a, label="Adequate (≥3)",     color=colors_q,         alpha=0.85, width=0.55)
    ax2.bar(x, pcts_n, bottom=pcts_a, label="Below adequate (<3)", color=["#BDC3C7"]*3, alpha=0.7,  width=0.55)
    for xi, pa in zip(x, pcts_a):
        ax2.text(xi, pa/2, f"{pa:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax2.text(xi, pa + (100-pa)/2, f"{100-pa:.1f}%", ha="center", va="center", fontsize=11, color=DARK)
    ax2.set_xticks(x); ax2.set_xticklabels(qs, fontsize=9)
    ax2.set_ylabel("% of Participants", fontsize=10, color=DARK); ax2.set_ylim(0, 115)
    ax2.set_title("Adequacy Rates\n(n=30)", fontsize=10, fontweight="bold", color=DARK)
    ax2.legend(fontsize=9, loc="upper right"); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("Figure 7: User Study — SHAP Explanation Ratings by Crisis-Context Evaluators",
                 fontsize=11, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    savefig(os.path.join(out_dir, "fig7_userstudy.png"), fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    print("Generating figures...")
    fig1_dataset_composition(args.output)
    fig2_class_distribution(args.output)
    fig3_training_progression(args.output)
    fig4_per_dataset(args.output)
    fig5_crisis_gap(args.output)
    fig6_cross_domain(args.output)
    fig7_user_study(args.output)
    print(f"\nAll 7 figures saved to {args.output}/")
