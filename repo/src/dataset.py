"""
dataset.py
----------
Load, preprocess, and combine the three source datasets into
the crisis & defense misinformation benchmark.

Sources:
  - CoAID     : COVID-19 health crisis (cuilimeng/CoAID on GitHub)
  - LIAR      : Political / government disinformation
  - FakeNewsNet-PolitiFact : Political fact-checked news
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

FAKE_LIAR_LABELS = {"false", "pants-fire", "barely-true"}
REAL_LIAR_LABELS = {"true", "mostly-true", "half-true"}

LIAR_COLUMNS = [
    "id", "veracity", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true", "false_count",
    "half_true", "mostly_true", "pants_fire", "context",
]


# ── CoAID ─────────────────────────────────────────────────────────────

def load_coaid(data_dir: str) -> pd.DataFrame:
    """Load all CoAID CSV files from *data_dir* and return a unified DataFrame."""
    dfs = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        fname = os.path.basename(fpath)
        label = 0 if "Fake" in fname else 1
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig", on_bad_lines="skip")
        except Exception:
            continue
        text_col = next(
            (c for c in ["content", "abstract", "title", "newstitle"] if c in df.columns),
            None,
        )
        if text_col is None:
            continue
        df["text"] = df[text_col].fillna("").astype(str)
        df["label"] = label
        df["label_name"] = "fake" if label == 0 else "real"
        df["crisis_type"] = "health_crisis"
        df["source"] = "CoAID"
        dfs.append(df[["text", "label", "label_name", "crisis_type", "source"]])
    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return result[result["text"].str.len() > 30].reset_index(drop=True)


# ── LIAR ──────────────────────────────────────────────────────────────

def load_liar(data_dir: str) -> pd.DataFrame:
    """Load LIAR train / test / valid splits and return binary-labeled DataFrame."""
    dfs = []
    for split in ["train", "test", "valid"]:
        fpath = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(
                fpath, sep="\t", header=None,
                names=LIAR_COLUMNS, on_bad_lines="skip",
            )
        except Exception:
            continue
        df["veracity"] = df["veracity"].astype(str).str.strip().str.lower()
        for subset, lbl in [
            (df[df["veracity"].isin(FAKE_LIAR_LABELS)], 0),
            (df[df["veracity"].isin(REAL_LIAR_LABELS)], 1),
        ]:
            subset = subset.copy()
            subset["text"] = subset["statement"].fillna("").astype(str)
            subset["label"] = lbl
            subset["label_name"] = "fake" if lbl == 0 else "real"
            subset["crisis_type"] = "political_defense"
            subset["source"] = "LIAR"
            dfs.append(subset[["text", "label", "label_name", "crisis_type", "source"]])
    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return result[result["text"].str.len() > 15].reset_index(drop=True)


# ── FakeNewsNet ───────────────────────────────────────────────────────

def load_fakenewsnet(data_dir: str, source: str = "politifact") -> pd.DataFrame:
    """Load PolitiFact fake / real CSVs from FakeNewsNet."""
    dfs = []
    for fname, lbl in [(f"{source}_fake", 0), (f"{source}_real", 1)]:
        fpath = os.path.join(data_dir, f"{fname}.csv")
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
        except Exception:
            continue
        if "title" not in df.columns:
            continue
        df["text"] = df["title"].fillna("").astype(str)
        df["label"] = lbl
        df["label_name"] = "fake" if lbl == 0 else "real"
        df["crisis_type"] = "political_defense"
        df["source"] = f"FakeNewsNet-{source.capitalize()}"
        dfs.append(df[["text", "label", "label_name", "crisis_type", "source"]])
    result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return result[result["text"].str.len() > 10].reset_index(drop=True)


# ── Combine and balance ───────────────────────────────────────────────

def balance_source(df: pd.DataFrame, max_per_class: int = 600) -> pd.DataFrame:
    """Balance classes within a source DataFrame."""
    n = min(
        len(df[df["label"] == 0]),
        len(df[df["label"] == 1]),
        max_per_class,
    )
    return pd.concat([
        df[df["label"] == 0].sample(n, random_state=42),
        df[df["label"] == 1].sample(n, random_state=42),
    ])


def build_benchmark(
    coaid_dir: str,
    liar_dir: str,
    fnn_dir: str,
    max_per_class: int = 600,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Build the combined crisis & defense benchmark from three sources.

    Parameters
    ----------
    coaid_dir    : directory containing CoAID CSV files
    liar_dir     : directory containing LIAR TSV files
    fnn_dir      : directory containing FakeNewsNet CSVs
    max_per_class: maximum samples per class per source (for balancing)
    output_path  : if given, save the benchmark CSV here

    Returns
    -------
    pd.DataFrame with columns: text, label, label_name, crisis_type, source
    """
    print("Loading CoAID...")
    coaid = balance_source(load_coaid(coaid_dir), max_per_class=400)

    print("Loading LIAR...")
    liar = balance_source(load_liar(liar_dir), max_per_class=max_per_class)

    print("Loading FakeNewsNet...")
    fnn = balance_source(load_fakenewsnet(fnn_dir, source="politifact"), max_per_class=400)

    combined = (
        pd.concat([coaid, liar, fnn], ignore_index=True)
        .pipe(lambda d: d[d["text"].str.len() > 15])
        .drop_duplicates(subset="text")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print(f"\nBenchmark: {len(combined)} samples")
    print(combined.groupby(["source", "label_name"]).size().to_string())

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"Saved → {output_path}")

    return combined
