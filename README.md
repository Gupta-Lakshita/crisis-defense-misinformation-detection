# Misinformation Detection in Crisis and Defense Scenarios

![Tests](https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection/actions/workflows/tests.yml/badge.svg)

**A benchmark study with cross-domain evaluation and explainability analysis.**

*Under review. Code, data, and results are publicly available for reproducibility.*

---

## Key Findings

| Finding | Result |
|---|---|
| General-domain model on COVID-19 health crisis content | **39–42% Macro-F1** (below random chance) |
| Crisis-domain fine-tuning recovery | **+50pp** on health crisis content |
| Health crisis detection (XLM-R fine-tuned) | **90.99% Macro-F1** |
| Political defense detection (XLM-R fine-tuned) | **62.46% Macro-F1** |
| Health crisis vs political defense gap | **28.5pp** (holds across both model families) |
| XLM-R zero-shot to fine-tuned gain | **+48.4pp** |
| SHAP explanations found meaningful by crisis evaluators (Q2) | **53.3%** (n=30) |
| SHAP explanations trusted for crisis decisions (Q3) | **6.7%** (n=30) |

---

## Dataset

14,971 total samples across six publicly available datasets.

| Source | Crisis Type | Samples | Role |
|---|---|---|---|
| CoAID | Health Crisis (COVID-19 news) | 666 | Crisis benchmark |
| COVID-Constraint-Twitter | Health Crisis (social media) | 2,400 | Crisis benchmark |
| LIAR-PLUS | Political / Government | 2,400 | Crisis benchmark |
| FakeNewsNet-PolitiFact | Political / Defense | 800 | Crisis benchmark |
| FakeOrReal-News (balanced) | General Political News | 2,400 | Benchmark + cross-domain |
| FakeOrReal-News (full) | General Political News | 6,305 | Cross-domain source only |

See [`data/README.md`](data/README.md) for download instructions.

---

## Models

Four TF-IDF baselines + XLM-RoBERTa (`xlm-roberta-base`, 278M params).

| Model | Macro-F1 | ROC-AUC |
|---|---|---|
| TF-IDF + Logistic Regression | 77.62% | — |
| TF-IDF + Linear SVM | 77.21% | — |
| TF-IDF + Random Forest | 73.17% | — |
| TF-IDF + Gradient Boosting | 74.81% | — |
| XLM-R Zero-Shot | 33.33% | 47.23% |
| **XLM-R Fine-Tuned (3 epochs)** | **81.70%** | **92.04%** |

XLM-R fine-tuning: AdamW, lr=2e-5, weight_decay=0.01, batch=32, max_len=128, 3 epochs on Kaggle T4 GPU (~55 min total).

---

## Explainability and User Study

SHAP LinearExplainer applied to TF-IDF + LR model (200 background samples).

User study with **n=30 participants**, the majority from crisis-relevant disciplines (public health, journalism and media studies, policy and security studies, emergency management), with a smaller number from adjacent technical and scientific backgrounds.

| Question | Mean / 5.0 | SD | Adequate (≥3) |
|---|---|---|---|
| Q1: Prediction plausibility | 3.52 | 0.66 | 90.0% |
| Q2: Explanation meaningfulness | 2.72 | 0.85 | 53.3% |
| Q3: Crisis trustworthiness | 1.93 | 0.65 | 6.7% |

Pearson r(Q2, Q3) = 0.603 (p<0.001). Cohen's kappa (binary adequacy) = -0.008.
15 of 16 participants who rated Q2 adequate did not rate Q3 adequate.

---

## Research Gaps Addressed

**RG1:** Cross-domain transfer from general political news to crisis content had not been quantified.
**RG2:** Health crisis and political defense detection had not been compared as separate classification problems.
**RG3:** General-purpose multilingual models had not been tested against a zero-shot crisis baseline.
**RG4:** SHAP explanations had been validated with journalists but not with crisis decision-makers.

---

## Repository Structure

```
crisis-defense-misinformation-detection/
├── src/
│   ├── preprocessing.py      # Dataset loading, cleaning, benchmark construction
│   ├── baselines.py          # TF-IDF models + SHAP analysis
│   ├── xlmr_finetune.py      # XLM-RoBERTa fine-tuning (Kaggle GPU)
│   └── cross_domain.py       # Cross-domain transfer experiment
├── scripts/
│   └── generate_figures.py   # Regenerate all 7 paper figures
├── data/
│   ├── README.md             # Dataset download instructions
│   └── user_study_n30.csv    # User study Likert ratings (n=30, anonymised)
├── results/
│   ├── final_results.json         # All model results
│   ├── cross_domain_results.json  # Cross-domain transfer results
│   ├── user_study_results.json    # User study statistics
│   └── figures/                   # 7 paper figures (PNG, 180 DPI)
│       ├── fig1_dataset.png
│       ├── fig2_class_dist.png
│       ├── fig3_training.png
│       ├── fig4_per_dataset.png
│       ├── fig5_crisis_gap.png
│       ├── fig6_crossdomain.png
│       └── fig7_userstudy.png
├── tests/
│   └── test_results.py       # 21 tests — all passing
├── .github/workflows/
│   └── tests.yml             # CI: runs on every push to main
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Gupta-Lakshita/crisis-defense-misinformation-detection
cd crisis-defense-misinformation-detection
pip install -r requirements.txt

# After placing raw datasets in data/raw/:
python src/preprocessing.py

# Run TF-IDF baselines + SHAP
python src/baselines.py --data data/crisis_defense_benchmark_v4.csv

# Cross-domain experiment
python src/cross_domain.py \
  --general_data data/raw/fakeorreal/fake_or_real_news.csv \
  --crisis_test  data/crisis_defense_benchmark_v4.csv

# Regenerate figures
python scripts/generate_figures.py

# Run tests
pytest tests/ -v
```

For XLM-R fine-tuning, upload `crisis_defense_benchmark_v4.csv` to Kaggle and run `src/xlmr_finetune.py` on a T4 GPU notebook.

---

## Citation

If you use this benchmark or code, please cite:

```
Gupta-Lakshita (2025). Misinformation Detection in Crisis and Defense Scenarios:
A Benchmark Study with Cross-Domain Evaluation and Explainability Analysis.
Under review. github.com/Gupta-Lakshita/crisis-defense-misinformation-detection
```

---

## References

- Jadhav et al. (2025) HEMT-Fake. *Frontiers in Artificial Intelligence* 8:1690616.
- Kukkar and Kaur (2025). *Expert Systems with Applications* 281:127751.
- Nwaiwu et al. (2025) X-FRAME. *Frontiers in Artificial Intelligence* 8:1523102.
- Conneau et al. (2020) XLM-RoBERTa. *ACL 2020*, 8440–8451.
- Cui and Lee (2020) CoAID. arXiv:2006.00885.
- Patwa et al. (2021) COVID-Constraint. *AAAI CONSTRAINT Workshop*, 21–29.
- Wang (2017) LIAR. *ACL 2017*, 422–426.
- Shu et al. (2020) FakeNewsNet. *Big Data* 8(3):171–188.
- Tikanmäki and Ruoslahti (2024). *Journal of Military Studies*. doi:10.2478/jms-2024-0009.
