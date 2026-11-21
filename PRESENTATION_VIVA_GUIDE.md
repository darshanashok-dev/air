# Air Quality Forecasting for Indian Cities: Presentation & Viva Defense Guide

A comprehensive slide deck outline, member speaking scripts, and anticipated viva defense Q&A for the **Semester 3 Kaggle Mini Project**.

---

## 1. 12-Slide Presentation Deck Outline

### Slide 1: Title & Project Overview
- **Header**: Air Quality Forecasting for Indian Cities (2015–2020)
- **Sub-header**: A Comparative Benchmark of Statistical, Tree-Based ML, and Recurrent Deep Learning Architectures
- **Metadata**: Semester 3 Kaggle Mini Project | Team of 4
- **Core Research Question**: *Which model family forecasts next-day and multi-day AQI most accurately, and does the winning paradigm shift across contrasting climatological and geographical regimes?*

---

### Slide 2: The Four Contrasting Urban Airsheds
- **Delhi (Indo-Gangetic Basin)**: Severe baseline pollution, extreme winter temperature inversions, agricultural residue burning shocks, and photochemically driven secondary aerosol formation.
- **Bengaluru (Deccan Plateau)**: Moderate/satisfactory baseline air quality, lower annual volatility, regular diurnal wind circulation.
- **Hyderabad (Central Plateau)**: Continuous 2015–2020 monitoring records, distinct post-monsoon pollution patterns, high day-to-day autocorrelation ($> 0.88$).
- **Mumbai (Coastal Megacity)**: Maritime boundary layer, strong diurnal sea-breeze / land-breeze ventilation, continuous valid series from May 2018 onwards.
- **Visual Reference**: [01_aqi_timeline_rolling.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/01_aqi_timeline_rolling.png) and [02_monthly_seasonality.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/02_monthly_seasonality.png).

---

### Slide 3: Zero-Leakage Methodology & Strict Time Split
- **The Golden Rule**: *Never use random splits or shuffling on time-series.*
- **Chronological Split Architecture**:
  - **Training Set (2015–2018)**: Historical baseline learning.
  - **Validation Set (2019)**: Hyperparameter tuning, early stopping, model selection.
  - **Test Set (2020)**: Out-of-sample holdout (183 continuous days).
  - **Pre-Lockdown Holdout (Jan 1 – Mar 24, 2020)**: 84 days evaluated separately to decouple normal atmospheric performance from pandemic shutdown effects.
- **Unified Evaluation Metrics**:
  - Primary Metric: **MAE** (Mean Absolute Error).
  - Secondary Metrics: **RMSE**, **MAPE** ($\text{denom} \ge 1.0$), and **CPCB Qualitative Bucket Accuracy** (Good $\to$ Severe).
- **Zero-Leakage Contract**: Scalers fit strictly on training set; all lag features use strictly backward-looking windows ($t-1, t-2, \dots$).

---

### Slide 4: Baseline Benchmarks & Stationarity Diagnostics
- **Why Baselines Matter**: In time-series, persistence is a formidable opponent. Without comparing against simple inertia, complex models can report misleadingly low errors.
- **Three Reference Heuristics**:
  1. *Naive ($t-1$)*: Tomorrow equals today ($\hat{y}_{t+1} = y_t$).
  2. *Seasonal Naive ($t-7$)*: Tomorrow equals same day last week ($\hat{y}_{t+1} = y_{t-6}$).
  3. *7-Day Rolling Mean*: Tomorrow equals weekly moving average.
- **Statistical Tests**:
  - **ADF & KPSS**: All raw series show mean-reversion around strong seasonal cycles; first differencing achieves strict stationarity ($p < 0.001$).
  - **ACF/PACF**: Strong lag-1 autocorrelation ($0.85–0.92$) confirming massive day-to-day persistence.
- **Visual Reference**: [06_baseline_forecasts.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/06_baseline_forecasts.png) and [08_acf_pacf_diagnostics.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/08_acf_pacf_diagnostics.png).

---

### Slide 5: Classical Statistical Modeling: Walk-Forward SARIMA
- **Model Formulation**: Order-tuned SARIMA $(p, d, q)(P, D, Q)_7$ using `pmdarima.auto_arima` AIC minimization.
  - Delhi: $\text{SARIMA}(1, 1, 2)$ | Bengaluru: $\text{SARIMA}(1, 1, 1)(0, 0, 1)_7$
  - Hyderabad: $\text{SARIMA}(0, 1, 3)$ | Mumbai: $\text{SARIMA}(3, 1, 1)$
- **Evaluation Mechanism**: Rolling-origin (walk-forward) 1-day step forecasting across 2020 out-of-sample test set.
- **Key Finding**: SARIMA improved Delhi over baseline (**28.09 vs. 29.57**), but failed to beat simple Naive persistence in Hyderabad (**11.24 vs. 9.09**) and Mumbai (**14.87 vs. 14.56**).
- **Visual Reference**: [09_sarima_forecasts.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/09_sarima_forecasts.png).

---

### Slide 6: Machine Learning: Prophet & XGBoost
- **Meta Prophet with Domain Shock Regressors**:
  - Decomposes series into trend, annual Fourier harmonics, and weekly seasonality.
  - Added custom historical Diwali calendar window ($-1$ to $+2$ days).
  - *Finding*: Pure calendar Prophet failed catastrophically during lockdown (Delhi MAE: **102.89**). Adding an autoregressive lag-1 regressor reduced error by 61% to **39.61**.
- **XGBoost Gradient Boosted Trees**:
  - Feature engineering: Backward lags ($t-1, t-2, t-3, t-7, t-14$), rolling mean/std (7 and 30 days), and cross-pollutant interactions ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$).
  - Atmospheric Interpretability: $PM_{2.5}$ lag-1 and AQI lag-1 contributed $>65\%$ of total split gain.
  - *Result*: Achieved project championship in Bengaluru (**MAE: 6.88**).
- **Visual Reference**: [10_prophet_components.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/10_prophet_components.png) and [12_xgboost_feature_importance.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/12_xgboost_feature_importance.png).

---

### Slide 7: Deep Learning: PyTorch LSTM Networks
- **Architecture**: 2-layer PyTorch LSTM (64 hidden units, dropout 0.2, linear readout layer, Smooth L1 Loss, Adam optimizer, early stopping on validation loss).
- **Two Sequence Configurations**:
  - *Univariate*: 14-day sequence window of historical AQI.
  - *Multivariate*: 14-day sequence window across 7 criteria pollutants ($AQI, PM_{2.5}, PM_{10}, NO_2, CO, SO_2, O_3$).
- **The Delhi Breakthrough**:
  - Multivariate LSTM achieved **MAE of 23.16** in Delhi—beating XGBoost (**26.58**), SARIMA (**28.09**), and Naive baseline (**29.57**).
  - *Why?* Recurrent cell memory gates tracked chemical precursor accumulation ($NO_2, SO_2 \to \text{secondary aerosol particulates}$) before they manifested in headline AQI spikes.
- **Visual Reference**: [15_lstm_learning_curves.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/15_lstm_learning_curves.png) and [16_lstm_forecasts.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/16_lstm_forecasts.png).

---

### Slide 8: Grand Model Benchmark Leaderboard
- **36 Evaluated Configurations across 4 Cities**:

| City | Champion Model | Full Test MAE | Baseline MAE | Improvement | CPCB Bucket Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Delhi** | **Multivariate LSTM** | **23.16** | 29.57 | **+21.7%** | **74.86%** |
| **Bengaluru** | **XGBoost (Multivariate Lags)** | **6.88** | 7.78 | **+11.6%** | **81.97%** |
| **Hyderabad** | **Naive ($t-1$) Baseline** | **9.09** | 9.09 | **+0.0%** | **86.89%** |
| **Mumbai** | **Naive ($t-1$) Baseline** | **14.56** | 14.56 | **+0.0%** | **81.97%** |

- **Core Conclusion**: There is no single "magic" architecture. The optimal model is determined by local atmospheric volatility and meteorological mechanics.
- **Visual Reference**: [18_cross_city_mae_comparison.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/18_cross_city_mae_comparison.png) and [22_model_performance_heatmap.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/22_model_performance_heatmap.png).

---

### Slide 9: Multi-Horizon Forecast Decay ($t+1, t+3, t+7$)
- **How rapidly does accuracy decay as lookahead expands?**
  - **Delhi (High Volatility)**: Naive MAE degrades by +71% ($29.57 \to 50.65$); Direct XGBoost jumps from $27.51$ to $91.60$ at $t+7$. Without forward-looking wind vectors and planetary boundary layer forecasts, multi-day predictions degenerate.
  - **Bengaluru (Moderate Inland)**: Direct XGBoost remains remarkably resilient, achieving **12.72 MAE at $t+7$ and outperforming the persistence baseline (13.36)**.
- **Takeaway**: Structured machine learning delivers actionable multi-day public health warnings in moderate climates, but volatile airsheds require coupled numerical weather prediction.
- **Visual Reference**: [19_horizon_decay_analysis.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/19_horizon_decay_analysis.png).

---

### Slide 10: The COVID-19 Structural Break Natural Experiment
- **The Event**: Nationwide lockdown enacted March 25, 2020.
- **The Atmospheric Impact**: Ambient AQI plunged across all centers:
  - Mumbai: **-56.7%** (149.5 $\to$ 64.7) | Delhi: **-49.4%** (248.1 $\to$ 125.5)
  - Hyderabad: **-32.3%** (94.8 $\to$ 64.2) | Bengaluru: **-31.6%** (96.3 $\to$ 65.8)
- **Model Robustness Comparison**:
  - *Pure Seasonal (Prophet)*: Suffered catastrophic failure (Delhi MAE: 102.89) due to blind seasonal priors.
  - *Autoregressive (SARIMA, XGBoost, LSTM)*: Experienced an initial 48-hour shock, but adapted back to low-error tracking within 2 days.
- **Visual Reference**: [20_lockdown_impact_analysis.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/20_lockdown_impact_analysis.png).

---

### Slide 11: Stratified Error Diagnostics & The Spike Blindspot
- **Residual Stratification by CPCB Severity Category (Delhi)**:
  - Satisfactory Air (AQI 51–100): XGBoost MAE is **15.11**.
  - Severe Air (AQI > 400): XGBoost MAE skyrockets to **48.33**.
- **The Critical Vulnerability**: All statistical and machine learning models inherently pull forecasts toward the historical conditional mean. Sudden non-stationary spikes (Diwali firecrackers, farm fires) are consistently under-predicted without external real-time satellite burn inputs.
- **Visual Reference**: [21_stratified_error_analysis.png](file:///home/d/study/sem3/kaggle/mini/eda_plots/21_stratified_error_analysis.png).

---

### Slide 12: Strategic Recommendations for Pollution Control Boards (CPCB/SPCB)
1. **Tiered Hybrid Architecture**: Deploy lightweight XGBoost/SARIMA models for routine daily forecasting (executes in milliseconds on standard edge servers); trigger Multivariate LSTM networks when winter inversion stagnation conditions are detected.
2. **Prioritize Real-Time Meteorological Sensor Telemetry**: Integrating Planetary Boundary Layer (PBL) height, wind velocity/direction vectors, and humidity is the single most critical enhancement needed to extend reliable forecasting to 3–7 day horizons.
3. **Dynamic Structural Break Detection**: Integrate automated change-point detection algorithms (e.g., CUSUM) to down-weight historical seasonal priors during unprecedented policy shifts, industrial shutdowns, or odd-even traffic rules.

---

## 2. Team Member Role Distribution & Speaking Scripts

### Member 1: Data Lead (EDA, Imputation, Correlation & Seasonality)
- **Slides Owned**: Slides 1, 2, and 3.
- **Key Talking Points**:
  - *"Our first priority was ensuring zero data leakage. We enforced a strict chronological split (2015–2018 Train, 2019 Validation, and 2020 Test) rather than random splits."*
  - *"In exploratory analysis, we identified that Delhi suffers from severe winter particulate inversions driven by $PM_{2.5}$ and $PM_{10}$, whereas Bengaluru exhibits stable diurnal variation. Because 2020 experienced the national lockdown, we isolated a pre-lockdown evaluation window to evaluate models fairly under normal conditions."*

### Member 2: Classical Statistical Modeling (Baselines & SARIMA)
- **Slides Owned**: Slides 4 and 5.
- **Key Talking Points**:
  - *"We established three rigorous persistence baselines. In environmental time-series, persistence is tough to beat because atmospheric conditions carry massive day-to-day inertia."*
  - *"Using ADF and KPSS tests, we confirmed first differencing yields strict stationarity. We tuned SARIMA orders using AIC/BIC minimization and evaluated models using walk-forward 1-day origin expansion. SARIMA improved Delhi over baseline, but in Hyderabad and Mumbai, simple persistence remained superior."*

### Member 3: Machine Learning Models (Prophet & XGBoost)
- **Slides Owned**: Slide 6 and Slide 11.
- **Key Talking Points**:
  - *"We evaluated Meta Prophet with custom holiday windows for Diwali. While pure calendar Prophet collapsed during lockdown, adding autoregressive lags stabilized performance."*
  - *"For XGBoost, we engineered strictly backward-looking lag and rolling features. XGBoost achieved the best overall computational efficiency across the project, winning outright in Bengaluru with an MAE of 6.88 and achieving 81.97% CPCB bucket accuracy."*

### Member 4: Deep Learning & Synthesis Lead (PyTorch LSTM, Horizon Decay & Conclusions)
- **Slides Owned**: Slides 7, 8, 9, 10, and 12.
- **Key Talking Points**:
  - *"We trained PyTorch LSTM networks in univariate and multivariate configurations. In Delhi, Multivariate LSTM achieved our project's lowest error (MAE: 23.16), proving that recurrent cells capture multi-pollutant precursor dynamics prior to headline spikes."*
  - *"However, deep learning is not a silver bullet. In moderate and coastal cities, it offered zero advantage over XGBoost or Naive persistence while requiring 20x more compute. We also proved that multi-day forecast accuracy decays steeply unless forward-looking meteorological data is coupled."*

---

## 3. Anticipated Viva Defense Q&A

### Q1: Why did the simple Naive persistence ($t-1$) baseline beat complex models in Hyderabad and Mumbai?
**Model Answer**:
> *"Air quality in steady-state regimes exhibits immense autocorrelation ($> 0.88$). In Hyderabad, day-to-day pollution is governed by continuous, stable background emissions with low variance, meaning yesterday's AQI is already an optimal conditional estimator. In Mumbai, maritime sea breezes cause rapid diurnal ventilation; without forward-looking wind speed and direction data, complex machine learning and neural architectures overfit to historical land patterns. As established in time-series literature, when variance is low or unobserved exogenous drivers dominate, complex models incur estimation variance that exceeds their bias reduction, making simple persistence hard to beat."*

### Q2: Why did Multivariate LSTM outperform XGBoost and SARIMA specifically in Delhi?
**Model Answer**:
> *"Delhi experiences extreme non-linear atmospheric chemistry during winter inversions. Secondary aerosol formation is a multi-step photochemical process where precursor gases like $NO_2, SO_2,$ and $CO$ accumulate over several days before converting into dangerous $PM_{2.5}$ particulates. The Multivariate LSTM's recurrent memory cells received 14-day continuous sequential trajectories across all 7 criteria pollutants, allowing the forget and input gates to recognize precursor accumulation patterns before the headline AQI spike materialized. XGBoost, while powerful, relied on summary rolling statistics that flattened fine-grained temporal sequence dynamics."*

### Q3: Why did pure seasonal Prophet fail so catastrophically during the 2020 COVID-19 lockdown?
**Model Answer**:
> *"Meta Prophet's standard formulation is a Generalized Additive Model: $y(t) = g(t) + s(t) + h(t) + \epsilon_t$. Its seasonal component $s(t)$ relies on fixed Fourier harmonics learned from historical pre-2020 years. When the nationwide lockdown occurred on March 25, 2020, anthropogenic emissions dropped by ~50% overnight. Because pure Prophet lacked autoregressive state feedback, it stubbornly predicted high historical pre-monsoon dust levels despite empty streets and closed factories. When we added an explicit autoregressive lag-1 regressor, Prophet adapted quickly, reducing Delhi MAE from 102.89 down to 39.61."*

### Q4: Why did you report both Full-Year 2020 and Pre-Lockdown Test metrics?
**Model Answer**:
> *"Reporting only the full 2020 test set would conflate model forecasting capability with an unprecedented structural break. Pre-lockdown (Jan 1 to Mar 24, 2020, 84 days) reflects true out-of-sample performance under standard economic and meteorological conditions. Reporting both allows us to demonstrate two vital properties: (1) baseline operational accuracy under normal emissions, and (2) structural break resilience when extreme real-world shocks disrupt historical priors."*

### Q5: How did you prevent data leakage during scaling and feature engineering?
**Model Answer**:
> *"We enforced three strict rules: First, our split was strictly chronological (Train 2015–2018, Val 2019, Test 2020) with zero random shuffling. Second, all MinMax scalers were fitted solely on the training partition; validation and test sets were transformed using the frozen training parameters. Third, all lag features ($t-1, t-2, \dots$) and rolling statistics were computed strictly backward-looking. For multi-horizon direct forecasting ($t+h$), features were lagged by at least $h$ days so no future information was accessible at forecast origin."*

### Q6: Why did forecast error rise so dramatically at the 7-day horizon ($t+7$)?
**Model Answer**:
> *"Air quality is an open thermodynamic system governed by chaotic synoptic meteorology. Local particulate concentration is heavily dictated by wind speed, direction, temperature inversions, and planetary boundary layer height. Statistical and machine learning models trained only on historical air quality measurements can exploit short-term atmospheric inertia up to 24–48 hours. Beyond that, local autocorrelation dissolves, and predicting AQI 7 days ahead without forward numerical weather prediction is physically impossible, causing error to regress toward the climatological mean."*

### Q7: If you were advising the Central Pollution Control Board (CPCB), which model would you recommend deploying?
**Model Answer**:
> *"We recommend a tiered, hybrid operational deployment: For standard regional monitoring in cities with moderate variance (like Bengaluru and Hyderabad), deploy XGBoost or SARIMA. They run in milliseconds, require no GPU infrastructure, and provide interpretable feature importances. For high-risk episodic airsheds in Northern India (like Delhi NCR), deploy Multivariate LSTM conditioned on real-time criteria pollutant telemetry. Furthermore, the board should invest in real-time integration of boundary layer meteorological data, which our study identified as the primary missing variable for multi-day reliability."*

### Q8: What are the primary limitations of your study and how would you extend it?
**Model Answer**:
> *"Our primary limitation was the absence of co-located meteorological telemetry (temperature, wind velocity, planetary boundary layer height) and real-time satellite fire radiative power data. Future extensions should incorporate: (1) ERA5 reanalysis weather data, (2) spatial graph neural networks (GNNs) to model cross-city transboundary pollution transport from Punjab to Delhi, and (3) conformal prediction intervals to provide calibrated uncertainty bounds for public health advisories."*
