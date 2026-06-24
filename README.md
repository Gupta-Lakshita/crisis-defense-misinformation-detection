# Crisis & Defense Misinformation Detection

A benchmark study for detecting misinformation in crisis and defense scenarios across four datasets (8,872 total samples), with cross-domain transfer analysis and SHAP explainability.

**Paper:** *Misinformation Detection in Crisis and Defense Scenarios: A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis* _(under review)_

![Tests](https://github.com/YOUR-USERNAME/crisis-defense-misinformation-detection/actions/workflows/tests.yml/badge.svg)

---

## Key Findings

**1. General-domain models collapse on crisis content.**
A TF-IDF + SVM model trained on 6,335 general political news articles scores only 45.46% Macro-F1 on COVID-19 health crisis content — below random chance. Crisis-domain fine-tuning recovers 38.0 percentage points. Domain adaptation is not optional.

**2. Health crisis and political disinformation are different problems.**
The same crisis-adapted model scores 83.48% F1 on health crisis content (CoAID) but only 60.48% on political disinformation (LIAR) — a 23-point gap that persists across all model types. These domains need different detection strategies.

**3. Zero-shot transformers provide no useful signal.**
XLM-RoBERTa without fine-tuning scores 34.11% Macro-F1. Three epochs of crisis-domain fine-tuning raises this to 69.81% (+35.7pp). A well-tuned linear SVM (71.55%) remains competitive at this dataset scale.

---

## Study Design

| Component | Details |
|---|---|
| Crisis benchmark | 2,537 samples — CoAID + LIAR + FakeNewsNet-PolitiFact |
| General domain (cross-domain) | 6,335 samples — FakeOrReal political news |
| **Total samples in study** | **8,872 across 4 datasets** |
| Crisis types | Health crisis (COVID-19) + Political / defense |
| Models evaluated | TF-IDF×4 baselines, XLM-R zero-shot, XLM-R fine-tuned |
| Experimental conditions | In-domain, cross-domain transfer (general→crisis, crisis→general) |

---

## Results

### Main Benchmark (Crisis-Adapted TF-IDF + SVM)

| Dataset | Domain | In-Domain F1 | General-Domain F1 | Adaptation Gain |
|---|---|---|---|---|
| CoAID | Health Crisis | **83.48%** | 45.46% | **+38.0pp** |
| FakeNewsNet-PolitiFact | Political / Defense | 79.98% | 62.67% | +17.3pp |
| LIAR | Political / Defense | 60.48% | 57.99% | +2.5pp |
| **Overall** | | **71.55%** | 57.03% | **+14.5pp** |

### All Models on Crisis Benchmark

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + LR | 69.70% | 69.70% | 69.60% | 69.60% | 77.50% |
| **TF-IDF + SVM** | **71.70%** | **71.70%** | 71.50% | **71.55%** | — |
| TF-IDF + RF | 65.70% | 65.70% | 65.70% | 65.70% | 75.30% |
| TF-IDF + GBM | 66.50% | 66.50% | 66.60% | 66.50% | 73.50% |
| XLM-R Zero-Shot | 51.77% | 25.89% | 50.00% | 34.11% | 52.95% |
| XLM-R Fine-Tuned (3 epochs) | 69.88% | 69.85% | 69.80% | 69.81% | 78.39% |

### XLM-R Fine-Tuning Progression

| Epoch | Val Accuracy | Val Macro-F1 | Val ROC-AUC |
|---|---|---|---|
| 0 (zero-shot) | 51.77% | 34.11% | 52.95% |
| 1 | 64.5% | 64.07% | 74.49% |
| 2 | 72.9% | 72.90% | 80.75% |
| 3 (best) | 73.9% | 73.88% | 81.97% |

---

## Datasets

| Source | Crisis Type | Samples | Role in Study |
|---|---|---|---|
| [CoAID](https://github.com/cuilimeng/CoAID) | Health Crisis (COVID-19) | 595 | Crisis benchmark |
| [LIAR](https://github.com/thiagorainmaker77/liar_dataset) | Political / Defense | 1,200 | Crisis benchmark |
| [FakeNewsNet-PolitiFact](https://github.com/KaiDMML/FakeNewsNet) | Political / Defense | 742 | Crisis benchmark |
| [FakeOrReal-News](https://github.com/lutzhamel/fake-news) | General Political News | 6,335 | Cross-domain baseline |
| **Total** | | **8,872** | |

---

## Repository Structure

```
crisis-defense-misinformation-detection/
├── src/
│   ├── dataset.py          # Data loading and preprocessing
│   ├── evaluate.py         # Evaluation metrics (per-source, per-type)
│   └── explain.py          # SHAP explainability
├── notebooks/
│   └── kaggle_xlmr.py      # XLM-RoBERTa fine-tuning (Kaggle/Colab)
├── scripts/
│   ├── build_dataset.py    # Download and merge source datasets
│   └── run_experiments.py  # Full baseline + cross-domain pipeline
├── data/
│   └── crisis_defense_benchmark_v2.csv   (crisis benchmark, 2,537 samples)
├── results/
│   └── figures/            # 12 paper figures
├── tests/
│   └── test_dataset.py
└── requirements.txt
```

---

## Reproducing Results

```bash
git clone https://github.com/YOUR-USERNAME/crisis-defense-misinformation-detection
cd crisis-defense-misinformation-detection
pip install -r requirements.txt

# Baseline + cross-domain experiments (CPU, ~5 min)
python scripts/run_experiments.py

# XLM-R fine-tuning (GPU required — use Kaggle T4)
# Upload data/crisis_defense_benchmark_v2.csv as a Kaggle dataset
# Run notebooks/kaggle_xlmr.py with GPU enabled (~30 min)
```

---

## Citation

```bibtex
@article{yourname2025crisis,
  title   = {Misinformation Detection in Crisis and Defense Scenarios:
             A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis},
  author  = {Your Name},
  journal = {Under Review},
  year    = {2025}
}
```

## License

Code: MIT. Data subject to original dataset licenses.
