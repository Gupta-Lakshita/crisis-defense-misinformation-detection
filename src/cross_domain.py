"""
cross_domain.py
───────────────
Cross-domain transfer experiment.
Trains TF-IDF + SVM on FakeOrReal-News (general domain) only,
evaluates on the full crisis benchmark test set.

Usage:
    python src/cross_domain.py \
        --general_data data/raw/fakeorreal/fake_or_real_news.csv \
        --crisis_test  data/crisis_defense_benchmark_v4.csv \
        --output       results/cross_domain_results.json
"""

import json, argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.preprocessing import clean_text, CRISIS_TYPE_MAP


def run_cross_domain(
    general_data_path: str,
    crisis_test_path: str,
    output_path: str = "results/cross_domain_results.json",
    seed: int = 42,
) -> dict:

    # Load and clean general-domain training source
    gdf = pd.read_csv(general_data_path)
    text_col = next(c for c in gdf.columns if c.lower() in ["text", "title"])
    label_col = next(c for c in gdf.columns if c.lower() == "label")
    gdf["text"] = gdf[text_col].apply(clean_text)
    gdf["label"] = gdf[label_col].map({"REAL": 1, "FAKE": 0, "real": 1, "fake": 0})
    gdf = gdf.dropna(subset=["text", "label"])
    gdf["label"] = gdf["label"].astype(int)

    X_gen_train, X_gen_val, y_gen_train, y_gen_val = train_test_split(
        gdf["text"], gdf["label"], test_size=0.2, stratify=gdf["label"], random_state=seed
    )
    print(f"General-domain train: {len(X_gen_train)} samples")

    # Train general-domain model
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
    clf = LinearSVC(C=1.0, dual=False, random_state=seed)
    clf.fit(vec.fit_transform(X_gen_train), y_gen_train)

    # Own-domain F1 (sanity check)
    own_f1 = f1_score(y_gen_val, clf.predict(vec.transform(X_gen_val)), average="macro")
    print(f"Own-domain val Macro-F1: {own_f1:.4f}")

    # Load crisis benchmark and evaluate per source
    cdf = pd.read_csv(crisis_test_path)
    _, cdf_test, _, _ = train_test_split(
        cdf, cdf["label"], test_size=0.2, stratify=cdf["label"], random_state=seed
    )

    results = {
        "general_domain_train_n": len(X_gen_train),
        "own_domain_val_f1": round(own_f1, 4),
        "per_source": {},
    }

    for source in cdf_test["source"].unique():
        sub = cdf_test[cdf_test["source"] == source]
        y_pred = clf.predict(vec.transform(sub["text"].apply(clean_text)))
        f1 = f1_score(sub["label"], y_pred, average="macro")
        results["per_source"][source] = {
            "general_domain_f1": round(f1, 4),
            "crisis_type": sub["crisis_type"].iloc[0],
            "n": len(sub),
        }
        print(f"  {source}: {f1:.4f} (n={len(sub)})")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCross-domain results saved: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_data", required=True)
    parser.add_argument("--crisis_test",  required=True)
    parser.add_argument("--output", default="results/cross_domain_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_cross_domain(args.general_data, args.crisis_test, args.output, args.seed)
