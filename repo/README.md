# Crisis & Defense Misinformation Detection

A benchmark study and classification framework for detecting misinformation in crisis and defense scenarios, using traditional ML baselines and fine-tuned multilingual transformers (XLM-RoBERTa).

**Paper:** *Misinformation Detection in Crisis and Defense Scenarios: A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis* _(under review)_

---

## Overview

Most misinformation detection research is built on civilian benchmarks — Reddit posts, entertainment gossip sites, general political news. This project asks a different question: how well do these approaches actually work when the domain is a public health emergency, a breaking news crisis, or state-sponsored political disinformation?

The answer, based on our cross-domain evaluation, is that performance drops significantly across crisis types. Health crisis misinformation (COVID-19) reaches 83.5% F1 with a linear SVM. Political disinformation from verified fact-checkers scores 60.5% — a 23-point gap using the same model and same approach. That gap is the finding.

---

## Results

### Baseline Models — Combined Crisis & Defense Benchmark

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 69.7% | 69.7% | 69.6% | 69.6% | 77.5% |
| TF-IDF + Linear SVM | **71.7%** | **71.7%** | 71.5% | **71.6%** | — |
| TF-IDF + Random Forest | 65.7% | 65.7% | 65.7% | 65.7% | 75.3% |
| TF-IDF + Gradient Boosting | 66.5% | 66.5% | 66.6% | 66.5% | 73.5% |
| XLM-R Zero-Shot | — | — | — | — | — |
| **XLM-R Fine-Tuned** | — | — | — | — | — |

_XLM-R results to be added after training run_

### Per-Source Breakdown (Best Baseline: Linear SVM)

| Dataset | Domain | F1 | Precision | Recall | Samples |
|---|---|---|---|---|---|
| CoAID | Health Crisis | 83.5% | 84.2% | 84.2% | 115 |
| FakeNewsNet-PolitiFact | Political/Defense | 80.0% | 81.7% | 80.1% | 152 |
| LIAR | Political/Defense | 60.5% | 60.6% | 60.5% | 241 |

---

## Dataset

The benchmark combines three publicly available labeled datasets:

| Source | Crisis Type | Samples | Label Source |
|---|---|---|---|
| [CoAID](https://github.com/cuilimeng/CoAID) | Health Crisis (COVID-19) | 666 | WHO / fact-checkers |
| [LIAR](https://github.com/thiagorainmaker77/liar_dataset) | Political / Government | 1,200 | PolitiFact 6-class |
| [FakeNewsNet-PolitiFact](https://github.com/KaiDMML/FakeNewsNet) | Political / Defense | 742 | PolitiFact |
| **Combined** | | **2,537** | |

The combined benchmark (`data/crisis_defense_benchmark_v2.csv`) is included in this repo. All source datasets are publicly available under their original licenses.

---

## Repository Structure

```
crisis-defense-misinformation-detection/
├── src/
│   ├── dataset.py          # Data loading and preprocessing
│   ├── train_baselines.py  # TF-IDF + sklearn models
│   ├── evaluate.py         # Evaluation metrics
│   └── explain.py          # SHAP explainability
├── notebooks/
│   └── kaggle_xlmr.py      # XLM-RoBERTa fine-tuning (Kaggle/Colab)
├── scripts/
│   ├── build_dataset.py    # Download + merge source datasets
│   └── run_experiments.py  # Full baseline experiment pipeline
├── data/
│   └── crisis_defense_benchmark_v2.csv
├── results/
│   └── figures/            # All paper figures
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/crisis-defense-misinformation-detection
cd crisis-defense-misinformation-detection
pip install -r requirements.txt
```

### Run baseline experiments

```bash
python scripts/run_experiments.py
```

This downloads source datasets, builds the combined benchmark, trains all four baseline models, generates figures, and saves results to `results/`.

### Run XLM-RoBERTa fine-tuning

Upload `data/crisis_defense_benchmark_v2.csv` to Kaggle as a dataset, then run `notebooks/kaggle_xlmr.py` in a Kaggle notebook with GPU enabled. Expected runtime: 25--35 minutes on a free T4.

---

## Explainability

SHAP token-level attribution is generated for the best-performing model. Key findings:

- Facebook-origin linguistic markers (`"on facebook"`, `"see more"`) correlate with fake labels in the health crisis domain
- Authoritative institutional terms (`"researchers"`, `"pandemic"`, `"testing"`) correlate with real credible content
- Political disinformation (LIAR) shows weaker lexical separation than health crisis content, consistent with the lower F1

---

## Requirements

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
shap>=0.44.0
matplotlib>=3.7.0
seaborn>=0.12.0
transformers>=4.33.0
torch>=2.0.0
datasets>=2.14.0
accelerate>=0.21.0
```

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@article{yourname2025crisis,
  title   = {Misinformation Detection in Crisis and Defense Scenarios:
             A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis},
  author  = {Your Name and Co-authors},
  journal = {Under Review},
  year    = {2025}
}
```

---

## License

Code: MIT License.
Data: subject to original dataset licenses (CoAID: CC BY 4.0, LIAR: research use, FakeNewsNet: research use).
