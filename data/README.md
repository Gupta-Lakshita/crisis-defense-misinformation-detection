# Data

The benchmark datasets are not distributed in this repo due to licensing constraints.
Place raw source files in `data/raw/<source>/` as described below, then run:

```bash
python scripts/build_benchmark.py
```

## Dataset Sources

| Dataset | Where to get it | Place in |
|---|---|---|
| CoAID | https://github.com/cuilimeng/CoAID | `data/raw/coaid/` |
| COVID-Constraint-Twitter | https://competitions.codalab.org/competitions/26655 | `data/raw/constraint/` |
| LIAR-PLUS | https://github.com/Tariq60/LIAR-PLUS | `data/raw/liar_plus/` |
| FakeNewsNet-PolitiFact | https://github.com/KaiDMML/FakeNewsNet | `data/raw/fakenewsnet/` |
| FakeOrReal-News | https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset | `data/raw/fakeorreal/` |

## Pre-built Files (included)

| File | Description |
|---|---|
| `user_study_n30.csv` | User study Likert ratings (n=30 participants, anonymised) |

## Benchmark Statistics

After running `build_benchmark.py` you should get:

- `crisis_defense_benchmark_v4.csv` — 8,666 samples (crisis benchmark)
  - CoAID: 666 | COVID-Twitter: 2,400 | LIAR-PLUS: 2,400 | FakeNewsNet: 800 | FakeOrReal: 2,400
  - 4,333 fake / 4,333 real (perfectly balanced)
- The full FakeOrReal-News corpus (6,305 samples) is used separately in cross-domain experiments.
