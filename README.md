# Air Quality Forecasting for Indian Cities (2015–2020)

A comprehensive time-series forecasting study comparing heuristic baselines, classical statistical methods, gradient-boosted trees, and recurrent deep learning architectures for daily Air Quality Index (AQI) prediction across contrasting Indian urban airsheds.

---

## 1. Problem Statement & Research Objectives
Given historical daily air-quality measurements across Indian metropolitan centers, forecast daily AQI for next-day ($t+1$) and multi-day lookahead horizons ($t+3$, $t+7$).
- **Core Research Question**: Which model family forecasts next-day and multi-day AQI most effectively, and does the optimal paradigm change across contrasting climatological and geographical regimes?
- **Key Real-World Challenges Evaluated**:
  - Catastrophic non-linear episodic spikes (winter temperature inversions, Diwali firecrackers, post-monsoon stubble burning).
  - The unprecedented structural break of the March 25, 2020 nationwide COVID-19 lockdown (-31% to -57% drop in ambient AQI).
  - Lookahead horizon degradation ($t+1$ to $t+7$).

---

## 2. Selected Cities & Atmospheric Profiles
1. **Delhi (Severe / Extreme Volatility)**: Continental northern plains; severe winter inversion layers, post-monsoon agricultural burning shocks, high baseline particulate burden.
2. **Bengaluru (Moderate / Inland Plateau)**: Southern Deccan plateau; cleaner ambient air, lower annual variance, steady diurnal circulation.
3. **Hyderabad (Smooth Persistence Regime)**: Central plateau; continuous 2015–2020 monitoring records, distinct post-monsoon pollution patterns, high day-to-day autocorrelation (> 0.88).
4. **Mumbai (Maritime Coastal Dynamics)**: Coastal mega-city; strong land-breeze sea-breeze ventilation, continuous valid monitoring series from May 2018 onwards.

---

## 3. Repository Structure & Notebook Sequence
```text
.
├── data/
│   └── city_day.csv          # Air Quality Data in India (2015-2020)
├── utils.py                  # Shared loader, strict time splits, metrics, and plotting
├── 01_eda.ipynb              # Notebook 01: Exploratory Data Analysis & visual insights
├── 02_baselines_sarima.ipynb # Notebook 02: Baselines (Naive, Seasonal Naive) & Walk-Forward SARIMA
├── 03_prophet_xgboost.ipynb  # Notebook 03: Meta Prophet (with Diwali shock) & XGBoost (lag features)
├── 04_lstm.ipynb             # Notebook 04: PyTorch Deep Learning (Univariate & Multivariate LSTM)
├── 05_comparison.ipynb       # Notebook 05: Grand Leaderboard, Horizon Decay, Lockdown & Synthesis
├── submission_kaggle.ipynb   # Complete monolithic, self-contained Kaggle submission notebook
├── eda_plots/                # 22 publication-grade figures (PNG, 200 DPI)
├── results/                  # Consolidated benchmark metric CSVs
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 4. Methodology & Strict Zero-Leakage Protocol
1. **Strict Time-Based Split**:
   - **Training Set**: 2015-01-01 to 2018-12-31 (Historical baseline)
   - **Validation Set**: 2019-01-01 to 2019-12-31 (Hyperparameter tuning & early stopping)
   - **Test Set (Out-of-Sample)**: 2020-01-01 to 2020-07-01 (183 continuous holdout days)
   - **Pre-Lockdown Sub-Test**: 2020-01-01 to 2020-03-24 (84 continuous days before national lockdown)
   - *Zero shuffling or random splitting permitted.*
2. **Unified Evaluation Metrics**:
   - **MAE** (Mean Absolute Error, Primary Metric)
   - **RMSE** (Root Mean Squared Error)
   - **MAPE** (Mean Absolute Percentage Error, protected with $\text{denom} \ge 1.0$)
   - **CPCB Bucket Accuracy** (Agreement with Central Pollution Control Board qualitative air categories: Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
3. **Zero Data Leakage Contract**:
   - Scalers fit strictly on training set.
   - All lag and rolling features computed using strictly backward-looking temporal windows ($t-1, t-2, \dots$).
4. **Principled Imputation**:
   - Short gaps ($\le 7$ days) linearly interpolated along time index.
   - Long gaps (> 7 days) left untouched; records trimmed after extensive unrecorded prefixes.

---

## 5. Grand Model Benchmark Leaderboard

Evaluated on identical 2020 out-of-sample holdout test sets across 36 city-model configurations:

| City | Model Family | Architecture / Order | Full Test MAE | Full Test RMSE | Full Test MAPE (%) | CPCB Bucket Acc (%) | Pre-Lockdown MAE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Delhi** | Heuristic Baseline | Naive ($t-1$) | 29.57 | 39.95 | 17.53% | 71.04% | 41.99 |
| **Delhi** | Classical Statistical | SARIMA(1, 1, 2) | 28.09 | 36.81 | 17.35% | 67.76% | 39.49 |
| **Delhi** | Additive (GAM) | Prophet (+ Lag-1 Regressor) | 39.61 | 46.79 | 28.61% | 63.93% | 41.60 |
| **Delhi** | Gradient Boosted Trees | XGBoost (Multivariate Lags) | 26.58 | 35.41 | 17.50% | 72.68% | 35.20 |
| **Delhi** | Deep Learning (PyTorch)| LSTM (Univariate, 14-Day) | 29.11 | 37.65 | 18.19% | 70.49% | 40.17 |
| **Delhi** | **Deep Learning (PyTorch)**| **LSTM (Multivariate, 7 Pollutants)** | **23.16** | **30.56** | **14.61%** | **74.86%** | **31.09** |
| | | | | | | | |
| **Bengaluru** | Heuristic Baseline | Naive ($t-1$) | 7.78 | 10.45 | 9.87% | 83.06% | 9.35 |
| **Bengaluru** | Classical Statistical | SARIMA(1, 1, 1)(0, 0, 1, 7) | 7.73 | 10.07 | 10.15% | 80.33% | 8.95 |
| **Bengaluru** | Additive (GAM) | Prophet (+ Lag-1 Regressor) | 15.41 | 18.37 | 21.28% | 69.40% | 15.57 |
| **Bengaluru** | **Gradient Boosted Trees** | **XGBoost (Multivariate Lags)** | **6.88** | **9.12** | **8.73%** | **81.97%** | **8.07** |
| **Bengaluru** | Deep Learning (PyTorch)| LSTM (Univariate, 14-Day) | 7.41 | 10.00 | 9.57% | 79.23% | 8.95 |
| **Bengaluru** | Deep Learning (PyTorch)| LSTM (Multivariate, 7 Pollutants) | 7.46 | 9.43 | 9.90% | 79.78% | 8.82 |
| | | | | | | | |
| **Hyderabad** | **Heuristic Baseline** | **Naive ($t-1$)** | **9.09** | **12.48** | **11.80%** | **86.89%** | **11.65** |
| **Hyderabad** | Classical Statistical | SARIMA(0, 1, 3) | 11.24 | 14.49 | 15.97% | 81.42% | 14.38 |
| **Hyderabad** | Additive (GAM) | Prophet (+ Lag-1 Regressor) | 10.94 | 14.26 | 15.66% | 80.87% | 13.09 |
| **Hyderabad** | Gradient Boosted Trees | XGBoost (Multivariate Lags) | 10.66 | 13.06 | 16.18% | 81.97% | 12.56 |
| **Hyderabad** | Deep Learning (PyTorch)| LSTM (Univariate, 14-Day) | 9.80 | 13.05 | 13.07% | 82.51% | 12.19 |
| **Hyderabad** | Deep Learning (PyTorch)| LSTM (Multivariate, 7 Pollutants) | 9.68 | 12.18 | 14.45% | 83.61% | 11.20 |
| | | | | | | | |
| **Mumbai** | **Heuristic Baseline** | **Naive ($t-1$)** | **14.56** | **23.80** | **12.75%** | **81.97%** | **24.25** |
| **Mumbai** | Classical Statistical | SARIMA(3, 1, 1) | 14.87 | 23.05 | 14.28% | 78.69% | 23.60 |
| **Mumbai** | Additive (GAM) | Prophet (+ Lag-1 Regressor) | 15.25 | 23.14 | 14.75% | 79.78% | 22.81 |
| **Mumbai** | Gradient Boosted Trees | XGBoost (Multivariate Lags) | 21.57 | 28.94 | 23.30% | 74.86% | 28.34 |
| **Mumbai** | Deep Learning (PyTorch)| LSTM (Univariate, 14-Day) | 23.63 | 31.32 | 24.17% | 73.22% | 35.94 |
| **Mumbai** | Deep Learning (PyTorch)| LSTM (Multivariate, 7 Pollutants) | 19.81 | 28.60 | 18.40% | 77.05% | 30.30 |

---

## 6. Multi-Horizon Forecast Decay Analysis ($t+1, t+3, t+7$)

Evaluating how quickly forecast accuracy decays as public health lookahead expands from next-day to 1-week warning:

| City | Architecture | $t+1$ MAE | $t+3$ MAE | $t+7$ MAE | 7-Day Error Decay |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Delhi** | Naive Persistence | 29.57 | 50.15 | 50.65 | +71.3% error increase |
| **Delhi** | Direct XGBoost | **27.51** | 60.87 | 91.60 | Rapid decay without wind/PBL data |
| **Bengaluru** | Naive Persistence | 7.78 | 12.55 | 13.36 | Moderate decay |
| **Bengaluru** | Direct XGBoost | **7.51** | **11.26** | **12.72** | **Beats Naive baseline at $t+7$** |
| **Hyderabad** | Naive Persistence | **9.09** | **18.52** | **23.69** | Strong persistence advantage |
| **Hyderabad** | Direct XGBoost | 10.22 | 22.15 | 27.29 | Stable bounded error |
| **Mumbai** | Naive Persistence | **14.56** | **26.61** | **28.22** | Coastal sea-breeze baseline |
| **Mumbai** | Direct XGBoost | 18.12 | 34.94 | 28.72 | Overfits short historical record |

---

## 7. Key Empirical Findings & Conclusions

1. **The Regional Divergence Theorem**:
   - **Delhi (Extreme Hotspot)**: Deep learning delivers immense value (**Multivariate LSTM MAE: 23.16**, +21.7% over baseline). Recurrent cells effectively track chemical precursor transformations ($NO_2, SO_2, CO \to \text{Secondary Particulates}$) during severe winter inversions.
   - **Bengaluru (Moderate Plateau)**: Machine learning dominates (**XGBoost MAE: 6.88**, +11.6% over baseline). Structured decision trees capture calendar harmonics and rolling statistics without deep neural overfitting.
   - **Hyderabad & Mumbai (Smooth Persistence & Maritime)**: The **Naive ($t-1$) baseline is virtually unbeatable** without real-time meteorological wind speed and planetary boundary layer height data.
2. **Computational Overhead vs. Performance Trade-off**:
   - PyTorch LSTM requires **20–30x more compute time**, GPU acceleration, sequence windowing overhead, and hyperparameter tuning compared to XGBoost (<3 seconds on CPU).
   - *Verdict*: Deep learning is **not** a silver bullet. It should be reserved exclusively for high-variance airsheds with severe photochemical interactions.
3. **The COVID-19 Structural Break Natural Experiment**:
   - Following the March 25, 2020 national lockdown, ambient AQI plummeted by **-31.6% (Bengaluru)** to **-56.7% (Mumbai)** and **-49.4% (Delhi)**.
   - Pure seasonal models (Prophet Pure Calendar) failed catastrophically (**Delhi MAE: 102.89**), predicting high pre-monsoon pollution despite halted traffic. Autoregressive models (SARIMA, XGBoost, LSTM) suffered an initial 48-hour transitional shock before adapting back to low-error tracking.
4. **The Spike Blindspot**:
   - Across all models, error scales sharply with pollution severity (in Delhi, XGBoost MAE rises from **15.11** during Satisfactory air to **48.33** during Severe episodes). Models inherently pull forecasts toward the conditional mean; predicting catastrophic sudden shocks requires forward-looking satellite fire and weather data.

---

## 8. Setup & Reproduction Instructions

```bash
# Clone and configure environment
git clone <repo_url>
cd mini

# Python 3.11+ virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the complete sequence top-to-bottom
# (Or open 01_eda.ipynb through 05_comparison.ipynb in Jupyter / VSCode)
```
