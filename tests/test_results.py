"""
Tests for crisis-defense-misinformation-detection
Runs in CI via .github/workflows/tests.yml
All tests use only the shipped results JSON files — no dataset downloads needed.
"""

import json, os
import pytest
import numpy as np


RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR       = os.path.join(os.path.dirname(__file__), "..", "data")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")


# ── Fixture helpers ────────────────────────────────────────────────────

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    assert os.path.exists(path), f"Missing: {path}"
    with open(path) as f:
        return json.load(f)


# ── 1. Benchmark integrity ─────────────────────────────────────────────

def test_benchmark_sample_counts():
    r = load_json("final_results.json")
    assert r["benchmark"]["total_samples"] == 14971
    assert r["benchmark"]["benchmark_samples"] == 8666
    assert r["benchmark"]["general_domain_samples"] == 6305


def test_benchmark_dataset_count():
    r = load_json("final_results.json")
    assert len(r["benchmark"]["datasets"]) == 6


def test_train_test_split():
    r = load_json("final_results.json")
    assert r["benchmark"]["test_samples"] == 1733
    assert r["benchmark"]["train_samples"] == 6933


# ── 2. Baseline results ────────────────────────────────────────────────

def test_lr_macro_f1():
    r = load_json("final_results.json")
    f1 = r["baseline_results"]["TF-IDF + Logistic Regression"]["macro_f1"]
    assert 0.77 <= f1 <= 0.78, f"LR F1 out of expected range: {f1}"


def test_svm_macro_f1():
    r = load_json("final_results.json")
    f1 = r["baseline_results"]["TF-IDF + Linear SVM"]["macro_f1"]
    assert 0.77 <= f1 <= 0.78, f"SVM F1 out of expected range: {f1}"


def test_baselines_present():
    r = load_json("final_results.json")
    expected = [
        "TF-IDF + Logistic Regression",
        "TF-IDF + Linear SVM",
        "TF-IDF + Random Forest",
        "TF-IDF + Gradient Boosting",
    ]
    for m in expected:
        assert m in r["baseline_results"], f"Missing baseline: {m}"


# ── 3. XLM-R results ──────────────────────────────────────────────────

def test_xlmr_finetuned_f1():
    r = load_json("final_results.json")
    f1 = r["xlmr_results"]["epoch_3_best"]["test_macro_f1"]
    assert f1 >= 0.81, f"XLM-R fine-tuned F1 below 0.81: {f1}"


def test_xlmr_finetuned_roc_auc():
    r = load_json("final_results.json")
    auc = r["xlmr_results"]["epoch_3_best"]["test_roc_auc"]
    assert auc >= 0.92, f"XLM-R ROC-AUC below 0.92: {auc}"


def test_xlmr_zero_shot_collapse():
    r = load_json("final_results.json")
    f1 = r["xlmr_results"]["zero_shot"]["macro_f1"]
    assert f1 < 0.40, f"Zero-shot F1 should be near random (0.33): {f1}"


def test_xlmr_finetuning_gain():
    r = load_json("final_results.json")
    gain = r["xlmr_results"]["epoch_3_best"]["zero_shot_to_finetuned_gain_pp"]
    assert gain >= 48.0, f"Fine-tuning gain < 48pp: {gain}"


# ── 4. Cross-domain collapse ───────────────────────────────────────────

def test_cross_domain_collapse_health():
    r = load_json("cross_domain_results.json")
    for source in ["CoAID", "COVID-Constraint-Twitter"]:
        f1 = r["results"][source]["general_domain_svm_f1"]
        assert f1 < 0.50, f"{source} general-domain F1 should be below random: {f1}"


def test_cross_domain_adaptation_gain():
    r = load_json("cross_domain_results.json")
    coaid_gain = r["results"]["CoAID"]["adaptation_gain_pp"]
    assert coaid_gain >= 45.0, f"CoAID adaptation gain < 45pp: {coaid_gain}"


# ── 5. Per-crisis-type gap ─────────────────────────────────────────────

def test_health_crisis_vs_political_defense_gap():
    r = load_json("final_results.json")
    health_f1 = r["per_crisis_type_results"]["health_crisis"]["xlmr_f1"]
    defense_f1 = r["per_crisis_type_results"]["political_defense"]["xlmr_f1"]
    gap = (health_f1 - defense_f1) * 100
    assert gap >= 28.0, f"Health-defense gap < 28pp: {gap:.1f}pp"


def test_liar_plus_ceiling():
    r = load_json("final_results.json")
    liar_f1 = r["per_dataset_results"]["LIAR-PLUS"]["xlmr_f1"]
    assert liar_f1 < 0.60, f"LIAR-PLUS XLM-R F1 unexpectedly high: {liar_f1}"


# ── 6. User study ─────────────────────────────────────────────────────

def test_user_study_n():
    r = load_json("user_study_results.json")
    assert r["n"] == 30


def test_user_study_q1_adequate():
    r = load_json("user_study_results.json")
    pct = r["results"]["Q1_plausibility"]["adequate_pct"]
    assert pct >= 85.0, f"Q1 adequate % below 85: {pct}"


def test_user_study_q3_low_trust():
    r = load_json("user_study_results.json")
    pct = r["results"]["Q3_crisis_trust"]["adequate_pct"]
    assert pct < 15.0, f"Q3 adequate % unexpectedly high: {pct}"


def test_user_study_q2_q3_gap():
    r = load_json("user_study_results.json")
    gap = r["statistics"]["q2_to_q3_gap_pp"]
    assert gap >= 45.0, f"Q2-Q3 gap < 45pp: {gap}"


# ── 7. Data files ─────────────────────────────────────────────────────

def test_user_study_csv_exists():
    path = os.path.join(DATA_DIR, "user_study_n30.csv")
    assert os.path.exists(path)


def test_user_study_csv_stats():
    import csv
    path = os.path.join(DATA_DIR, "user_study_n30.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    q1 = [float(r["q1_plausibility"]) for r in rows]
    assert abs(np.mean(q1) - 3.52) < 0.05


# ── 8. Figures ────────────────────────────────────────────────────────

def test_all_figures_exist():
    expected = [f"fig{i}_{name}.png" for i, name in enumerate(
        ["dataset", "class_dist", "training", "per_dataset",
         "crisis_gap", "crossdomain", "userstudy"], 1
    )]
    for fname in expected:
        path = os.path.join(FIGURES_DIR, fname)
        assert os.path.exists(path), f"Missing figure: {fname}"
