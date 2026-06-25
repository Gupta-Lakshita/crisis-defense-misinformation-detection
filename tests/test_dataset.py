"""Tests for the crisis & defense misinformation detection benchmark."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from src.evaluate import compute_metrics


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 100.0
    assert m["macro_f1"] == 100.0


def test_compute_metrics_random():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    m = compute_metrics(y_true, y_pred)
    assert 0 <= m["macro_f1"] <= 100
    assert 0 <= m["accuracy"] <= 100


def test_benchmark_v4_exists():
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'crisis_defense_benchmark_v4.csv')
    assert os.path.exists(path), "Benchmark v4 CSV not found"
    df = pd.read_csv(path)
    assert len(df) > 8000
    for col in ['text', 'label', 'source', 'crisis_type', 'label_name']:
        assert col in df.columns, f"Missing column: {col}"
    assert set(df['label'].unique()) == {0, 1}


def test_benchmark_v4_balance():
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'crisis_defense_benchmark_v4.csv')
    df = pd.read_csv(path)
    counts = df['label'].value_counts()
    ratio = counts.min() / counts.max()
    assert ratio >= 0.9, f"Dataset too imbalanced: {dict(counts)}"


def test_benchmark_v4_sources():
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'crisis_defense_benchmark_v4.csv')
    df = pd.read_csv(path)
    expected_sources = {'CoAID', 'COVID-Constraint-Twitter',
                        'LIAR-PLUS', 'FakeNewsNet-PolitiFact', 'FakeOrReal-News'}
    actual_sources   = set(df['source'].unique())
    assert expected_sources == actual_sources, \
        f"Source mismatch. Got: {actual_sources}"


def test_benchmark_v2_exists():
    path = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'crisis_defense_benchmark_v2.csv')
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) > 2000
