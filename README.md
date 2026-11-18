# Air Quality Forecasting for Indian Cities (2015–2020)

A time-series forecasting project comparing statistical, machine learning, and deep learning models for daily Air Quality Index (AQI) prediction across contrasting Indian cities.

## Problem Statement
Given historical daily air-quality measurements for Indian cities, forecast daily AQI for next-day ($t+1$) and multi-day horizons ($t+3$, $t+7$). Evaluate whether classical statistical, tree-based ML, or recurrent deep learning architectures perform best across diverse pollution profiles.

## Selected Cities
- **Delhi**: Severe winter spikes, strong annual seasonality, high baseline pollution.
- **Bengaluru**: Inland plateau, moderate-to-clean air, lower annual volatility.
- **Hyderabad**: Multi-year continuous recordings (2015–2020), distinct post-monsoon pollution patterns.
- **Mumbai**: Coastal dynamics, continuous valid AQI series available from May 2018 onwards.

## Project Structure
```text
.
├── data/
│   └── city_day.csv          # Air Quality Data in India (2015-2020)
├── utils.py                  # Shared loader, time-based split, metrics, and plotting
├── 01_eda.ipynb              # Exploratory Data Analysis & visual insights
├── 02_baselines_sarima.ipynb # Baselines (Naive, Seasonal Naive) & SARIMA
├── 03_prophet_xgboost.ipynb  # Prophet (with Diwali event) & XGBoost (lag features)
├── 04_lstm.ipynb             # PyTorch LSTM (univariate & multivariate)
├── 05_comparison.ipynb       # Cross-city comparison, error analysis & conclusions
├── requirements.txt          # Python dependencies
└── README.md
```

## Methodology & Strict Rules
1. **Strict Time-Based Split**:
   - **Train**: 2015–2018
   - **Validation**: 2019
   - **Test**: 2020 (with separate pre-lockdown evaluation for Jan 1 to Mar 24, 2020)
   - *Never use random splits or shuffling.*
2. **Unified Evaluation**: All models evaluated on identical test dates using shared functions in `utils.py`:
   - Mean Absolute Error (MAE, primary metric)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
   - CPCB AQI Bucket Accuracy
3. **Zero Data Leakage**:
   - Scalers fit only on training data.
   - Lag and rolling features computed using strictly backward-looking windows.
4. **Principled Imputation**:
   - Short gaps ($\le 7$ days) linearly interpolated.
   - Long gaps (> 7 days) left untouched; series trimmed to start after extensive unrecorded prefixes.

## Setup & Installation
```bash
# Using Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
