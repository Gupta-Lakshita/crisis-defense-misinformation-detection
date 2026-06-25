"""
explain.py
----------
SHAP-based explainability for TF-IDF + linear models.
Generates token-level attribution plots for crisis misinformation detection.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. Run: pip install shap")


def run_shap(pipeline, X_test, out_path: str = None, max_background: int = 200):
    """
    Run SHAP LinearExplainer on a TF-IDF + LogisticRegression pipeline.

    Parameters
    ----------
    pipeline    : fitted sklearn Pipeline with 'tfidf' and 'clf' steps
    X_test      : array-like of test texts
    out_path    : if given, save figure here
    max_background: number of background samples for SHAP

    Returns
    -------
    dict with 'fake_words' and 'real_words' lists of (word, shap_value)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required. Install with: pip install shap")

    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    X_transformed = vec.transform(X_test)

    n_bg = min(X_transformed.shape[0], max_background)
    explainer = shap.LinearExplainer(
        clf, X_transformed[:n_bg], feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X_transformed)
    feature_names = vec.get_feature_names_out()
    mean_signed = shap_values.mean(axis=0)

    # Tokens that push toward fake (negative SHAP for label=0)
    fake_idx = np.argsort(mean_signed)[:20]
    real_idx = np.argsort(mean_signed)[::-1][:20]

    fake_words = [
        (feature_names[i], abs(mean_signed[i]))
        for i in fake_idx if len(feature_names[i]) > 2
    ][:15]

    real_words = [
        (feature_names[i], mean_signed[i])
        for i in real_idx if len(feature_names[i]) > 2
    ][:15]

    if out_path:
        _plot_shap(fake_words, real_words, out_path)

    return {"fake_words": fake_words, "real_words": real_words}


def _plot_shap(fake_words, real_words, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, words, color, title in [
        (axes[0], fake_words, "#EF5350",
         "Top Tokens: FAKE Misinformation\n(Crisis & Defense Benchmark)"),
        (axes[1], real_words, "#43A047",
         "Top Tokens: REAL Credible Content\n(Crisis & Defense Benchmark)"),
    ]:
        names_ = [w for w, v in words]
        vals_  = [v for w, v in words]
        ax.barh(names_[::-1], vals_[::-1], color=color, alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        "SHAP Explainability — Crisis & Defense Misinformation Detection\n"
        "Token-level attribution across health crisis + political defense domains",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP figure → {out_path}")
