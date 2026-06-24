# Crisis & Defense Misinformation Detection

A benchmark study and classification framework for detecting misinformation in crisis and defense scenarios, using traditional ML baselines and fine-tuned multilingual transformers (XLM-RoBERTa).

**Paper:** *Misinformation Detection in Crisis and Defense Scenarios: A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis* _(under review)_

![Tests](https://github.com/YOUR-USERNAME/crisis-defense-misinformation-detection/actions/workflows/tests.yml/badge.svg)

---

## Key Findings

**1. Domain adaptation is not optional.** Zero-shot XLM-RoBERTa scores 34.11% Macro-F1 on the crisis benchmark. Three epochs of fine-tuning on crisis-domain data brings it to 69.81% — a 35.7-point gain. A general-purpose multilingual model cannot be deployed as-is in a crisis or defense context.

**2. Crisis types are not interchangeable.** Health crisis content (CoAID, COVID-19) reaches 83.4% F1 with both the SVM baseline and fine-tuned XLM-R. Political disinformation (LIAR) scores 55.6–60.5% — a persistent 17–28 point gap across all models. Different deployment contexts need different detection strategies.

**3. Traditional ML stays competitive on small datasets.** TF-IDF + Linear SVM (71.55% F1) marginally outperforms fine-tuned XLM-R (69.81%) on this 2,537-sample benchmark. The transformer advantage requires more domain-specific training data, which is a practical constraint in defense and crisis contexts where labeled data is scarce.

---

## Results

### Full Model Comparison

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 69.70% | 69.70% | 69.60% | 69.60% | 77.50% |
| **TF-IDF + Linear SVM** | **71.70%** | **71.70%** | 71.50% | **71.55%** | — |
| TF-IDF + Random Forest | 65.70% | 65.70% | 65.70% | 65.70% | 75.30% |
| TF-IDF + Gradient Boosting | 66.50% | 66.50% | 66.60% | 66.50% | 73.50% |
| XLM-R Zero-Shot | 51.77% | 25.89% | 50.00% | 34.11% | 52.95% |
| XLM-R Fine-Tuned (3 epochs) | 69.88% | 69.85% | 69.80% | 69.81% | 78.39% |

### Per-Dataset Breakdown

| Dataset | Domain | SVM F1 | XLM-R F1 | n (test) |
|---|---|---|---|---|
| CoAID | Health Crisis | 83.5% | 83.43% | 115 |
| FakeNewsNet-PolitiFact | Political/Defense | 80.0% | 81.95% | 152 |
| LIAR | Political/Defense | 60.5% | 55.60% | 241 |

### XLM-R Fine-Tuning Progression

| Epoch | Val Accuracy | Val Macro-F1 | Val ROC-AUC |
|---|---|---|---|
| 0 (zero-shot) | 51.77% | 34.11% | 52.95% |
| 1 | 64.5% | 64.07% | 74.49% |
| 2 | 72.9% | 72.90% | 80.75% |
| 3 | 73.9% | 73.88% | 81.97% |

---

## Dataset

| Source | Crisis Type | Samples | Label Source |
|---|---|---|---|
| [CoAID](https://github.com/cuilimeng/CoAID) | Health Crisis (COVID-19) | 595 | WHO / fact-checkers |
| [LIAR](https://github.com/thiagorainmaker77/liar_dataset) | Political / Government | 1,200 | PolitiFact 6-class |
| [FakeNewsNet-PolitiFact](https://github.com/KaiDMML/FakeNewsNet) | Political / Defense | 742 | PolitiFact |
| **Combined** | | **2,537** | |

---

## Repository Structure

```
crisis-defense-misinformation-detection/
├── src/
│   ├── dataset.py          # Data loading and preprocessing
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
│   └── figures/            # All paper figures (10 total)
├── tests/
│   └── test_dataset.py
└── requirements.txt
```

---

## Setup and Reproduction

```bash
git clone https://github.com/YOUR-USERNAME/crisis-defense-misinformation-detection
cd crisis-defense-misinformation-detection
pip install -r requirements.txt

# Run all baseline experiments
python scripts/run_experiments.py

# XLM-R fine-tuning (requires GPU — run on Kaggle or Colab)
# Upload data/crisis_defense_benchmark_v2.csv to Kaggle as a dataset
# then run notebooks/kaggle_xlmr.py with GPU T4 enabled
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

Code: MIT. Data subject to original dataset licenses (CoAID: CC BY 4.0, LIAR: research use, FakeNewsNet: research use).
