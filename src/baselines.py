"""
baselines.py
────────────
TF-IDF baseline models: Logistic Regression, Linear SVM, Random Forest, Gradient Boosting.
Includes SHAP explainability for the LR model.

Usage:
    python src/baselines.py --data data/crisis_defense_benchmark_v4.csv \
                            --output results/baseline_results.json
"""

import json, argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import shap


MODELS = {
    "TF-IDF + Logistic Regression": {
        "vectorizer": TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True),
        "clf": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42),
    },
    "TF-IDF + Linear SVM": {
        "vectorizer": TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True),
        "clf": LinearSVC(C=1.0, dual=False, random_state=42),
    },
    "TF-IDF + Random Forest": {
        "vectorizer": TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True),
        "clf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    },
    "TF-IDF + Gradient Boosting": {
        "vectorizer": TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True),
        "clf": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    },
}


def evaluate(y_true, y_pred, model_name: str) -> dict:
    return {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def run_baselines(
    data_path: str = "data/crisis_defense_benchmark_v4.csv",
    output_path: str = "results/baseline_results.json",
    shap_output: str = "results/shap_values.npy",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:

    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} samples (fake={sum(df.label==0)}, real={sum(df.label==1)})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, stratify=df["label"], random_state=seed
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    all_results = {}
    lr_pipeline = None

    for name, config in MODELS.items():
        print(f"\nTraining {name}...")
        vec = config["vectorizer"]
        clf = config["clf"]

        X_tr = vec.fit_transform(X_train)
        X_te = vec.transform(X_test)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        result = evaluate(y_test, y_pred, name)
        all_results[name] = result
        print(f"  Macro-F1: {result['macro_f1']:.4f}  Accuracy: {result['accuracy']:.4f}")

        if "Logistic" in name:
            lr_pipeline = (vec, clf, X_tr)

    # SHAP analysis on LR model
    if lr_pipeline and shap_output:
        print("\nRunning SHAP LinearExplainer on LR model...")
        vec, clf, X_tr = lr_pipeline
        background = shap.sample(X_tr, 200)
        explainer = shap.LinearExplainer(clf, background)
        X_te_shap = vec.transform(X_test[:500])
        shap_values = explainer.shap_values(X_te_shap)
        np.save(shap_output, shap_values)

        # Top tokens per class
        feature_names = np.array(vec.get_feature_names_out())
        mean_shap = np.mean(shap_values, axis=0)
        top_fake_idx = np.argsort(mean_shap)[::-1][:20]
        top_real_idx = np.argsort(mean_shap)[:20]

        all_results["shap_top_fake_tokens"] = feature_names[top_fake_idx].tolist()
        all_results["shap_top_real_tokens"] = feature_names[top_real_idx].tolist()
        print(f"  Top fake tokens: {all_results['shap_top_fake_tokens'][:5]}")
        print(f"  Top real tokens: {all_results['shap_top_real_tokens'][:5]}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/crisis_defense_benchmark_v4.csv")
    parser.add_argument("--output", default="results/baseline_results.json")
    parser.add_argument("--shap_output", default="results/shap_values.npy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baselines(args.data, args.output, args.shap_output, seed=args.seed)
