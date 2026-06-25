# Crisis & Defense Misinformation Detection

A benchmark study evaluating misinformation detection across crisis and defense scenarios. Uses 14,971 samples from six datasets, with cross-domain transfer experiments and SHAP explainability analysis.

**Paper:** *Misinformation Detection in Crisis and Defense Scenarios: A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis*

**Author:** Gupta-Lakshita | [GitHub](https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection)

![Tests](https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection/actions/workflows/tests.yml/badge.svg)

> **Note on badge:** The badge activates after the first push to `main`. If it shows "not found", push the repo and wait ~2 minutes for the workflow to complete.

---

## Key Findings

**1. General-domain models collapse on crisis content.**
A model trained on 6,305 general political news samples scores 39.11% Macro-F1 on COVID-19 health crisis tweets — 11 points below random chance. Crisis-domain adaptation recovers up to 50 percentage points.

**2. Health crisis and political defense are different problems.**
The same crisis-adapted model scores 90.99% F1 on health crisis content but only 62.46% on political disinformation — a 28.5-point gap that holds across both TF-IDF and transformer model families.

**3. Transformers need enough in-domain data.**
XLM-RoBERTa zero-shot scores 33.33% Macro-F1. Fine-tuning on 8,666 crisis-specific samples brings this to 81.70%, surpassing all TF-IDF baselines (+4.1pp). The gap between zero-shot and fine-tuned is 48.4 percentage points.

**4. SHAP explanations do not transfer to crisis decision-makers.**
83.3% of evaluators found the model predictions plausible. Only 25.0% found the SHAP explanation words meaningful, and 16.7% would trust them for a crisis decision. This 57-point drop from the 82% meaningfulness rating reported for journalist evaluators in HEMT-Fake (Jadhav et al., 2025) confirms that explainability outputs need calibration for operational crisis contexts.

---

## Study Design

| Component | Details |
|---|---|
| Crisis benchmark | 8,666 samples — 5 datasets |
| General domain (cross-domain only) | 6,305 samples — FakeOrReal-News |
| **Total samples in study** | **14,971 across 6 datasets** |
| Crisis types covered | Health crisis (COVID-19) + Political / defense |
| Models evaluated | TF-IDF × 4, XLM-R zero-shot, XLM-R fine-tuned |
| Explainability | SHAP token attribution + user study (n=12) |

---

## Full Results

### All Models on Crisis Benchmark (n=1,733 test)

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 77.62% | 77.62% | 77.62% | 77.62% | — |
| TF-IDF + Linear SVM | 77.22% | 77.21% | 77.22% | 77.21% | — |
| XLM-R Zero-Shot | 50.00% | 25.00% | 50.00% | 33.33% | 47.23% |
| **XLM-R Fine-Tuned (3 epochs)** | **81.78%** | **82.30%** | **81.78%** | **81.70%** | **92.04%** |

### Per-Dataset Breakdown

| Dataset | Domain | SVM F1 | XLM-R F1 | n (test) |
|---|---|---|---|---|
| FakeOrReal-News | General Political | 85.2% | 95.69% | 466 |
| CoAID | Health Crisis (News) | 88.4% | 92.68% | 138 |
| COVID-Constraint-Twitter | Health Crisis (Social Media) | 89.0% | 90.50% | 484 |
| FakeNewsNet-PolitiFact | Political / Defense | 74.8% | 83.59% | 153 |
| LIAR-PLUS | Political / Defense | 55.3% | 55.33% | 493 |

### Cross-Domain Transfer

| Dataset | Domain | Crisis-Adapted | General-Domain | Gain |
|---|---|---|---|---|
| FakeOrReal-News | General Political | 85.2% | 99.36% | N/A (own domain) |
| CoAID | Health Crisis (News) | 88.4% | 41.98% | +46.4pp |
| COVID-Constraint-Twitter | Health Crisis (Social Media) | 89.0% | **39.11%** | +49.9pp |
| FakeNewsNet-PolitiFact | Political / Defense | 74.8% | 59.89% | +14.9pp |
| LIAR-PLUS | Political / Defense | 55.3% | 49.33% | +5.9pp |

### User Study (n=12, SHAP Explanations)

| Question | Mean | % Adequate (≥3) |
|---|---|---|
| Q1 — Prediction plausibility | 3.00 / 5.0 | 83.3% |
| Q2 — Explanation meaningfulness | 2.17 / 5.0 | 25.0% |
| Q3 — Crisis trustworthiness | 1.83 / 5.0 | 16.7% |

---

## Datasets

| Source | Crisis Type | Samples | Role |
|---|---|---|---|
| [CoAID](https://github.com/cuilimeng/CoAID) | Health Crisis — COVID-19 News | 666 | Crisis benchmark |
| [COVID-Constraint-Twitter](https://github.com/diptamath/covid_fake_news) | Health Crisis — COVID-19 Social Media | 2,400 | Crisis benchmark |
| [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) | Political / Government | 2,400 | Crisis benchmark |
| [FakeNewsNet-PolitiFact](https://github.com/KaiDMML/FakeNewsNet) | Political / Defense | 800 | Crisis benchmark |
| [FakeOrReal-News](https://github.com/lutzhamel/fake-news) | General Political News | 6,305 | Cross-domain source |
| **Total** | | **14,971** | |

Both benchmark CSVs are included in `data/`.

---

## Repository Structure

```
crisis-defense-misinformation-detection/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── evaluate.py         # Evaluation metrics
│   └── explain.py          # SHAP explainability
├── scripts/
│   ├── build_dataset.py    # Download and merge source datasets
│   └── run_experiments.py  # Full baseline + cross-domain pipeline
├── notebooks/
│   └── kaggle_xlmr.py      # XLM-RoBERTa fine-tuning (run on Kaggle T4)
├── data/
│   ├── crisis_defense_benchmark_v2.csv   (original 3-dataset, 2,537 samples)
│   └── crisis_defense_benchmark_v4.csv   (expanded 5-dataset, 8,666 samples)
├── results/
│   ├── figures/            (28 paper-ready figures and tables)
│   ├── final_study_results.json
│   └── cross_domain_results.json
├── tests/
│   └── test_dataset.py
├── .github/
│   └── workflows/
│       └── tests.yml       (runs on every push; badge activates after first run)
└── requirements.txt
```

---

## Reproducing Results

```bash
git clone https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection
cd crisis-defense-misinformation-detection
pip install -r requirements.txt

# Baseline + cross-domain experiments (CPU only, ~10 min)
python scripts/run_experiments.py

# XLM-R fine-tuning (needs GPU — use Kaggle free T4)
# 1. Upload data/crisis_defense_benchmark_v4.csv to Kaggle as a dataset
# 2. Create new Kaggle notebook with GPU T4 enabled
# 3. Paste notebooks/kaggle_xlmr.py into a code cell
# 4. Update DATA_PATH to your dataset name
# 5. Run All (~30 min)
```

---

## Citation

```bibtex
@article{guptalakshita2025crisis,
  title   = {Misinformation Detection in Crisis and Defense Scenarios:
             A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis},
  author  = {Gupta-Lakshita},
  journal = {Under Review},
  year    = {2025},
  url     = {https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection}
}
```

## License

Code: MIT License. Data files subject to original dataset licenses — CoAID (CC BY 4.0), LIAR-PLUS (research use), FakeNewsNet (research use), COVID-Constraint (research use), FakeOrReal-News (MIT).
