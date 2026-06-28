"""
preprocessing.py
────────────────
Loads, cleans, balances, and merges the six benchmark datasets.

Datasets required (place in data/raw/):
  - coaid/         CoAID COVID-19 Healthcare Misinformation Dataset
  - constraint/    COVID-19 Fake News Detection (AAAI 2021 shared task)
  - liar_plus/     LIAR-PLUS dataset
  - fakenewsnet/   FakeNewsNet PolitiFact CSVs
  - fakeorreal/    FakeOrReal-News dataset

Usage:
    python src/preprocessing.py --output data/crisis_defense_benchmark_v4.csv
"""

import os, re, unicodedata, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


LABEL_MAP = {
    # LIAR-PLUS six-class to binary
    "true": 1, "mostly-true": 1, "half-true": 1,
    "false": 0, "pants-fire": 0, "barely-true": 0,
    # Standard binary
    "real": 1, "fake": 0, 1: 1, 0: 0,
}

CRISIS_TYPE_MAP = {
    "coaid": "health_crisis",
    "constraint": "health_crisis",
    "liar_plus": "political_defense",
    "fakenewsnet": "political_defense",
    "fakeorreal": "general_political",
}


def clean_text(text: str) -> str:
    """Lowercase, NFC-normalise, strip HTML and URLs, remove short entries."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[''\""]", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def balance_source(df: pd.DataFrame, n_per_class: int = None, seed: int = 42) -> pd.DataFrame:
    """Stratified subsample to equal fake/real counts."""
    counts = df["label"].value_counts()
    n = n_per_class if n_per_class else counts.min()
    return (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), n), random_state=seed))
          .reset_index(drop=True)
    )


def load_coaid(data_dir: str) -> pd.DataFrame:
    """Load CoAID (CSV format with 'content' and 'label' columns)."""
    frames = []
    for fname in ["news_fake.csv", "news_real.csv",
                  "social_fake.csv", "social_real.csv"]:
        path = os.path.join(data_dir, "coaid", fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        label = 0 if "fake" in fname else 1
        text_col = next((c for c in df.columns if c in ["content", "text", "title"]), None)
        if text_col:
            frames.append(pd.DataFrame({"text": df[text_col], "label": label}))
    if not frames:
        raise FileNotFoundError("CoAID files not found in data/raw/coaid/")
    return pd.concat(frames, ignore_index=True)


def load_constraint(data_dir: str) -> pd.DataFrame:
    """Load COVID-Constraint-Twitter (train.csv + val.csv)."""
    frames = []
    for fname in ["train.csv", "val.csv", "Constraint_Train.csv", "Constraint_Val.csv"]:
        path = os.path.join(data_dir, "constraint", fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        text_col = next((c for c in df.columns if c.lower() in ["tweet", "text"]), None)
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if text_col and label_col:
            df["label"] = df[label_col].map({"real": 1, "fake": 0})
            frames.append(df[["text" if text_col == "text" else text_col, "label"]]
                          .rename(columns={text_col: "text"}))
    if not frames:
        raise FileNotFoundError("Constraint files not found in data/raw/constraint/")
    return pd.concat(frames, ignore_index=True)


def load_liar_plus(data_dir: str) -> pd.DataFrame:
    """Load LIAR-PLUS (train2.tsv / test2.tsv / val2.tsv)."""
    frames = []
    for fname in ["train2.tsv", "test2.tsv", "val2.tsv"]:
        path = os.path.join(data_dir, "liar_plus", fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t", header=None, on_bad_lines="skip")
        # Col 2 = label, col 3 = statement, col 14 = justification
        if df.shape[1] >= 14:
            df["text"] = df[3].astype(str) + " " + df[14].astype(str)
        else:
            df["text"] = df[3].astype(str)
        df["label"] = df[2].str.lower().map(LABEL_MAP)
        frames.append(df[["text", "label"]].dropna())
    if not frames:
        raise FileNotFoundError("LIAR-PLUS files not found in data/raw/liar_plus/")
    return pd.concat(frames, ignore_index=True)


def load_fakenewsnet(data_dir: str) -> pd.DataFrame:
    """Load FakeNewsNet-PolitiFact (politifact_fake.csv / politifact_real.csv)."""
    frames = []
    for fname, label in [("politifact_fake.csv", 0), ("politifact_real.csv", 1)]:
        path = os.path.join(data_dir, "fakenewsnet", fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        text_col = next((c for c in df.columns if c.lower() in ["title", "news_url", "text"]), None)
        if text_col:
            frames.append(pd.DataFrame({"text": df[text_col], "label": label}))
    if not frames:
        raise FileNotFoundError("FakeNewsNet files not found in data/raw/fakenewsnet/")
    return pd.concat(frames, ignore_index=True)


def load_fakeorreal(data_dir: str) -> pd.DataFrame:
    """Load FakeOrReal-News (fake_or_real_news.csv)."""
    for fname in ["fake_or_real_news.csv", "news.csv"]:
        path = os.path.join(data_dir, "fakeorreal", fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            text_col = next((c for c in df.columns if c.lower() in ["text", "title"]), None)
            label_col = next((c for c in df.columns if c.lower() == "label"), None)
            if text_col and label_col:
                df["label"] = df[label_col].map({"REAL": 1, "FAKE": 0, "real": 1, "fake": 0})
                return df[["text", "label"]].rename(columns={text_col: "text"}).dropna()
    raise FileNotFoundError("FakeOrReal-News not found in data/raw/fakeorreal/")


def build_benchmark(
    data_dir: str = "data/raw",
    output_path: str = "data/crisis_defense_benchmark_v4.csv",
    balanced_n: int = 2400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds the full 8,666-sample crisis benchmark plus general domain source.
    Returns the combined DataFrame and writes to output_path.
    """
    loaders = {
        "coaid":       (load_coaid,       balanced_n // 4),   # 600
        "constraint":  (load_constraint,  balanced_n),        # 2400
        "liar_plus":   (load_liar_plus,   balanced_n),        # 2400
        "fakenewsnet": (load_fakenewsnet, balanced_n // 3),   # 800
        "fakeorreal":  (load_fakeorreal,  balanced_n),        # 2400
    }

    parts = []
    for name, (loader_fn, n) in loaders.items():
        try:
            df = loader_fn(data_dir)
            df["text"] = df["text"].apply(clean_text)
            df = df[df["text"].str.len() >= 15].drop_duplicates(subset=["text"])
            df = balance_source(df, n_per_class=n // 2, seed=seed)
            df["source"] = name
            df["crisis_type"] = CRISIS_TYPE_MAP[name]
            parts.append(df)
            print(f"  {name}: {len(df)} samples (fake={sum(df.label==0)}, real={sum(df.label==1)})")
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping {name}")

    if not parts:
        raise RuntimeError("No datasets loaded. Check data/raw/ directory.")

    combined = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    combined.to_csv(output_path, index=False)
    print(f"\nBenchmark saved: {output_path} ({len(combined)} samples)")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--output", default="data/crisis_defense_benchmark_v4.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_benchmark(args.data_dir, args.output, seed=args.seed)
