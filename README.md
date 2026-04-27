# Real Estate Investment Prediction from Demographic Trends

**Ansh Patel** — [github.com/anshcpatel11](https://github.com/anshcpatel11)

## What this does

Predicts which US ZIP codes are likely to see high real estate growth in the next year by combining Zillow housing data with US Census demographic data, engineering features from both sources, and training classifiers to flag high-growth areas.

A ZIP code is labeled "high growth" if Zillow's 1-year forecast puts it in the top 25% nationally. This project compares a simple weighted demographic index (baseline) against Logistic Regression and Random Forest to test whether ML adds real value. It does — the baseline barely beats a coin flip while Random Forest hits ~80% accuracy with a 0.87 AUC.

## Data

Two data sources are used:

**Zillow** — ZHVI (Zillow Home Value Index) has monthly home values for ~26k ZIP codes going back to 2000. ZHVF (Zillow Home Value Forecast) has Zillow's predicted growth percentages. Both CSVs go in `dataset/current/` and `dataset/future/` respectively. You can download them from [Zillow Research](https://www.zillow.com/research/data/).

**US Census ACS** — Pulled live through the Census API (you need an API key). Tables DP02, DP03, DP04, DP05 cover demographics, economics, housing characteristics, and social data for all ZIP codes from 2019 to 2023.

## Repo structure

```
├── dataset/                    <- Zillow CSVs go here (gitignored, download manually)
│   ├── current/                <- ZHVI csv
│   └── future/                 <- ZHVF csv
├── data/
│   └── output/                 <- Generated CSVs (features, train, test splits)
├── plots/                      <- Saved visualizations (auto-created by results.ipynb)
├── __init__.py                 <- Marks repo as a Python package
├── data_pipeline.py            <- Fetches Census data, cleans, engineers features, outputs train/test
├── baseline_scoring.py         <- Weighted demographic index baseline with CV support
├── supervised_models.py        <- Logistic Regression + Random Forest with printed results
├── tune_baseline_weights.py    <- Grid search over baseline weight scales and threshold
├── validate_models.py          <- Full CV comparison against dummy baselines
├── results.ipynb               <- ROC curves, confusion matrices, feature importance, geographic breakdown
├── .env.example                <- Environment variable template
└── requirements.txt
```

## Setup

Python 3.10+

```
pip install -r requirements.txt
```

**Environment variables** — copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
CENSUS_API_KEY=your_key_here          # free at https://api.census.gov/data/key_signup.html
ZILLOW_DATASET_DIR=/path/to/dataset   # folder containing current/ and future/ subdirs
```

The Zillow CSVs need to be downloaded manually from [Zillow Research](https://www.zillow.com/research/data/) and placed in the `dataset/` folders since they're too large for git.

## How to Run

**Step 1 — Build the feature set** (pulls Census API data, takes ~2 min):

```bash
python data_pipeline.py
```

Outputs three files to `data/output/`: `features.csv`, `train.csv`, `test.csv`.

**Step 2 — Run the demographic baseline:**

```bash
python baseline_scoring.py
```

**Step 3 — Train and evaluate ML models:**

```bash
python supervised_models.py
```

**Step 4 — Full CV validation against dummy baselines:**

```bash
python validate_models.py
```

**Step 5 — Generate all visualizations:**

Open and run `results.ipynb` to produce ROC curves, confusion matrices, feature importance charts, and a geographic breakdown of predicted high-growth ZIP codes. Plots are saved to `plots/`.

Optional — tune baseline weights via grid search:

```bash
python tune_baseline_weights.py
```

All scripts accept `--train` and `--test` flags if your CSVs are in a different location.

## Features

23 features engineered across two sources, 8 from Zillow and 15 from Census.

Zillow features capture housing market momentum: current home value, price returns over 3m/12m/36m/60m windows, an acceleration metric (short vs long term momentum), 12-month volatility, and how a ZIP compares to its metro area median.

Census features cover demographics and economics: population, median age, income, employment rate, education levels, housing vacancy, owner-occupancy rates, rent, home values, etc. Raw counts are converted to proportions where applicable.

## Results

|                           | Accuracy | Precision | Recall | F1    | AUC   |
|---------------------------|----------|-----------|--------|-------|-------|
| Baseline (weighted index) | 0.559    | 0.161     | 0.149  | 0.154 | —     |
| Logistic Regression       | 0.760    | 0.538     | 0.797  | 0.643 | 0.847 |
| Random Forest             | 0.796    | 0.597     | 0.755  | 0.667 | 0.872 |

The demographic index baseline z-score normalizes Census features and computes a weighted sum with positive weights for income, education, and young population, and negative weights for vacancy and elderly population, using the 75th percentile as a cutoff. It is essentially useless. Both ML models significantly outperform it, with Random Forest doing slightly better across the board.

## Design Notes

- Moved from county-level to ZIP-level analysis to capture intra-county variation in housing trends
- Missing values filled with column medians to retain rural/sparse ZIP codes rather than dropping them
- Label threshold (top 25% growth) is percentile-based so it stays stable across different dataset snapshots

## References

- [Zillow Research Data](https://www.zillow.com/research/data/)
- [US Census Bureau ACS API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- scikit-learn: Pedregosa et al., JMLR 12, 2011