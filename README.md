# infrastructure-reliability-analytics

Predicting the impact of environmental conditions on Scotland's railway reliability using machine learning. Built as part of an MSc Data Engineering dissertation at Glasgow Caledonian University, rebuilt in 2026 with corrected data pipelines, a broader model evaluation framework, and a proper test suite.

The core question this project tries to answer: **can weather data alone predict when train services are likely to be disrupted?** The results are more nuanced than a simple yes or no — KNN and the regularised regression models show meaningful predictive signal, while tree-based ensemble methods overfit badly on a dataset of this size. The analysis documents both what works and what's missing.

📄 [Full Analysis Report](docs/Infrastructure_Reliability_Report.docx)

---

## Structure

```
infrastructure-reliability-analytics/
│
├── data/
│   ├── raw/                              # original source files (not committed — see below)
│   └── processed/
│       ├── cleaned_delay_data.csv
│       └── cleaned_environmental_data.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_visualization.ipynb
│   └── 03_modelling.ipynb
│
├── src/
│   ├── logger.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── models.py
│
├── tests/
│   └── test_pipeline.py
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── environmental_timeseries.png
│   ├── cancellations_by_quarter.png
│   ├── wind_vs_cancellations.png
│   ├── seasonal_cancellations_boxplot.png
│   ├── model_rmse_comparison.png
│   ├── model_r2_comparison.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── model_evaluation_metrics.csv
│
├── docs/
│   └── Infrastructure_Reliability_Report.docx
│
├── .github/workflows/
│   └── ci.yml
├── .gitignore
├── Dockerfile
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Data sources

| Dataset | Source | Coverage |
|---|---|---|
| Train performance metrics | [Office of Rail and Road (ORR)](https://www.orr.gov.uk/statistics) | 2014–2026, quarterly |
| Weather data | [Met Office](https://www.metoffice.gov.uk) + [Visual Crossing](https://www.visualcrossing.com) | 2014–2026, monthly |

The ORR dataset includes quarterly cancellation scores broken down by fault category — infrastructure, operator, and external. The environmental dataset covers temperature (max/min/avg), precipitation, humidity, wind gust, wind speed, visibility, and cloud cover across Scotland.

After merging on a shared quarter key, the final dataset contains **45 quarterly observations** spanning the study period.

> Raw data files are not committed to this repo due to licensing. The cleaned processed files are included in `data/processed/`.

---

## Setup

**With pip:**

```bash
git clone https://github.com/herdaybusy/infrastructure-reliability-analytics.git
cd infrastructure-reliability-analytics

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**With conda:**

```bash
conda env create -f environment.yml
conda activate rail-analytics
```

---

## Running it

**Notebooks (run in order):**

```bash
jupyter notebook
```

1. `notebooks/01_data_preprocessing.ipynb`
2. `notebooks/02_data_visualization.ipynb`
3. `notebooks/03_modelling.ipynb`

**Scripts:**

```bash
python src/preprocessing.py    # cleans raw data → data/processed/
python src/visualization.py    # generates EDA charts → outputs/
python src/models.py           # trains models, saves metrics + charts → outputs/
```

**Tests:**

```bash
pytest tests/ -v
```

The test suite covers preprocessing logic, the merge, model outputs, and checks that all expected output files exist after the scripts have run.

---

## How it works

```
Raw data (ORR quarterly + Met Office/Visual Crossing monthly)
    ↓
Preprocessing — cleaning, renaming, type conversion, monthly resampling
    ↓
Quarterly aggregation — monthly env data averaged to quarterly means
    ↓
Merge on quarter key → 45 observations
    ↓
Feature selection — 9 environmental variables
StandardScaler + 80/20 train-test split (random_state=42)
36 training rows / 9 test rows
    ↓
Train 6 models:
    Linear Regression · Lasso (max_iter=10000) · Ridge
    Decision Tree · Random Forest · KNN
    ↓
Evaluate: MAE · MSE · RMSE · R²
```

The reason for aggregating environmental data to quarterly before merging is that the ORR publishes performance stats quarterly, not monthly. Doing the aggregation first avoids a granularity mismatch that would inflate the apparent size of the merged dataset.

---

## Results

| Rank | Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | **K-Nearest Neighbors** | 433.98 | 334,007.73 | **577.93** | **0.1583** |
| 2 | Ridge Regression | 518.43 | 368,144.49 | 606.75 | 0.0723 |
| 3 | Lasso Regression | 514.76 | 369,403.56 | 607.79 | 0.0691 |
| 4 | Linear Regression | 519.25 | 371,883.06 | 609.82 | 0.0629 |
| 5 | Random Forest | 670.14 | 916,693.59 | 957.44 | -1.3101 |
| 6 | Decision Tree | 885.30 | 1,371,096.76 | 1170.94 | -2.4552 |

KNN achieved the best performance with RMSE of 577.93 and a positive R² of 0.1583 — explaining roughly 16% of variance in quarterly cancellation scores from weather data alone. Four of the six models returned positive R² values. Random Forest and Decision Tree both returned negative R², consistent with known overfitting behaviour on datasets of this size.

**Top environmental predictors (Random Forest feature importance):**
- Cloud cover — strongest predictor (0.269)
- Humidity — second (0.215)
- Precipitation — third (0.154)

At quarterly granularity, sustained cloud cover and high humidity are better proxies for prolonged poor weather than peak wind readings, which average out across a three-month period.

---

## Key findings

- **45 quarterly observations** after merging — larger than the original 2024 dataset
- **KNN is the best model**, outperforming all regression and ensemble methods
- **4 out of 6 models return positive R²** — meaningful predictive signal exists in environmental data
- **Tree-based models overfit** on a dataset this small — Decision Tree is the worst performer by a wide margin
- **Cloud cover and humidity are the top predictors**, not wind gust as initially expected
- Winter quarters (Q1, Q4) consistently show the highest cancellation scores across all years in the study period
- Storm events (Doris, Isha, Jocelyn) are visible as clear outlier spikes in the time series

---

## Sample outputs

**Correlation heatmap**

![correlation heatmap](outputs/correlation_heatmap.png)

**Cancellations by quarter**

![cancellations by quarter](outputs/cancellations_by_quarter.png)

**RMSE comparison across models**

![rmse comparison](outputs/model_rmse_comparison.png)

**Feature importance — Random Forest**

![feature importance](outputs/feature_importance.png)

---

## What changed from the 2024 version

| Issue | 2024 | 2026 |
|---|---|---|
| File paths | Hardcoded Windows absolute paths — broke on any other machine | `os.path.join` with relative paths throughout |
| Data merging | Monthly env data merged without aggregating first | Correctly aggregated to quarterly before merge |
| Dataset size | ~30 rows | 45 rows after correct merge |
| Output files | No `plt.savefig` — charts only rendered in notebook | All charts saved to `outputs/` automatically |
| Models tested | 4 | 6 (added Lasso and Ridge) |
| Random seeds | Not set | `random_state=42` everywhere |
| Warnings | seaborn FutureWarnings, Lasso ConvergenceWarning | All fixed |
| Code structure | Notebooks only | Notebooks + `src/` scripts + logger + test suite |
| Best model | Linear Regression (negative R²) | KNN (R² = 0.1583) |

---

## Logging

All scripts use Python's standard `logging` module via `src/logger.py`. Key steps are logged at `INFO` level with timestamps:

```
2026-03-27 11:22:01,215 - INFO - __main__ - merged shape: (45, 19)
```

---

## Running tests

```bash
pytest tests/ -v                          # everything
pytest tests/ -v -k "env or delay"        # data validation only
pytest tests/ -v -k "model or scaler"     # model tests only
pytest tests/ -v -k "output or metrics"   # check output files exist
```

Output file tests skip gracefully on a fresh clone before scripts have been run.

---

## Future work

- **Richer data** — Network Rail infrastructure fault logs and rolling stock maintenance records would be the single biggest improvement. A formal ORR data request is the route to pursue this.
- **Finer temporal resolution** — monthly rather than quarterly observations would reduce overfitting risk and give ensemble methods more to learn from.
- **Hyperparameter tuning** — KNN with `n_neighbors=5` is the current best; a grid search could push performance further.
- **LSTM networks** — the time series structure makes this a reasonable candidate for sequence modelling to capture lagged weather effects on infrastructure.
- **Line-level granularity** — regional weather averages hide the difference between an exposed coastal viaduct and a sheltered urban route.

---

## Project context

Built as part of the **MSc Data Engineering** programme at **Glasgow Caledonian University**. The findings support active UK government policy areas — the Williams-Shapps Plan for Rail, the net zero 2050 transport strategy, and the Levelling Up connectivity agenda. Full policy discussion in the [report](docs/Infrastructure_Reliability_Report.docx).

---

## License

MIT

---

## Author

**Ahmed Adebisi**
MSc Data Engineering, Glasgow Caledonian University
[LinkedIn](https://www.linkedin.com/in/ahmed-adebisi-1a1576231) · [GitHub](https://github.com/Herdaybusy/Infrastructure_Reliability_Analytics)