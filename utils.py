"""
utils.py: Shared Time-Series Utilities for AQI Forecasting in Indian Cities.

Contains shared functions for:
1. Data loading and continuous daily reindexing.
2. Gap handling and short-gap interpolation (limit <= 7 days).
3. Strict time-based train/validation/test splitting (no shuffling, no future peeking).
4. Shared evaluation metrics (MAE, RMSE, MAPE, and AQI Bucket Accuracy).
5. Forecast visualization helpers with pre-lockdown markers.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CPCB (Central Pollution Control Board) Official AQI Category Buckets
AQI_BUCKETS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, np.inf, "Severe"),
]


def load_data(filepath: str = "data/city_day.csv") -> pd.DataFrame:
    """
    Loads daily air quality dataset from CSV with parsed Date column.
    
    Parameters
    ----------
    filepath : str
        Relative or absolute path to city_day.csv.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe sorted chronologically by Date.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values(by=["City", "Date"]).reset_index(drop=True)
    return df


def get_city_series(
    df: pd.DataFrame,
    city: str,
    target_col: str = "AQI",
    max_gap: int = 7,
    trim_leading_nans: bool = True
) -> pd.Series:
    """
    Extracts a strictly continuous daily time series for a specified city and target column.
    
    Enforces regular daily calendar frequency ('D').
    Trims long unrecorded leading periods (e.g. Mumbai pre-May 2018).
    Imputes short gaps up to max_gap days (default 7 days) via linear interpolation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing 'City', 'Date', and target column.
    city : str
        Target city name (e.g., 'Delhi', 'Bengaluru', 'Hyderabad', 'Mumbai').
    target_col : str
        Column to extract (default 'AQI').
    max_gap : int
        Maximum consecutive missing days to interpolate (Rule 4: max 7 days).
    trim_leading_nans : bool
        If True, trims any leading NaN streak before the first valid observation.
        
    Returns
    -------
    pd.Series
        Cleaned, continuous daily pandas Series with DatetimeIndex.
    """
    city_df = df[df["City"].str.lower() == city.lower()].copy()
    if city_df.empty:
        raise ValueError(f"City '{city}' not found in dataset.")
        
    series = city_df.set_index("Date")[target_col].sort_index()
    
    # Enforce continuous daily frequency (exposes missing calendar days)
    series = series.asfreq("D")
    
    # Trim leading NaNs to avoid carrying long initial unrecorded gaps
    if trim_leading_nans and series.notna().any():
        first_valid_date = series.first_valid_index()
        series = series.loc[first_valid_date:]
        
    # Impute only short gaps (max_gap consecutive days)
    if max_gap > 0:
        series = series.interpolate(method="time", limit=max_gap)
        
    series.name = f"{city}_{target_col}"
    return series


def get_city_dataframe(
    df: pd.DataFrame,
    city: str,
    columns: Optional[list] = None,
    max_gap: int = 7,
    trim_leading_nans: bool = True
) -> pd.DataFrame:
    """
    Extracts a continuous daily DataFrame with multiple features/pollutants for a city.
    Useful for multivariate models (XGBoost with previous pollutants, multivariate LSTM).
    """
    city_df = df[df["City"].str.lower() == city.lower()].copy()
    if city_df.empty:
        raise ValueError(f"City '{city}' not found in dataset.")
        
    city_df = city_df.set_index("Date").sort_index()
    if columns is not None:
        city_df = city_df[columns]
    else:
        # Exclude metadata columns
        drop_cols = [c for c in ["City", "AQI_Bucket"] if c in city_df.columns]
        city_df = city_df.drop(columns=drop_cols)
        
    # Continuous daily frequency
    city_df = city_df.asfreq("D")
    
    if trim_leading_nans and "AQI" in city_df.columns and city_df["AQI"].notna().any():
        first_valid_date = city_df["AQI"].first_valid_index()
        city_df = city_df.loc[first_valid_date:]
        
    if max_gap > 0:
        city_df = city_df.interpolate(method="time", limit=max_gap)
        
    return city_df


def split_time_series(
    data: Union[pd.Series, pd.DataFrame],
    train_end: str = "2018-12-31",
    val_end: str = "2019-12-31",
    lockdown_start: str = "2020-03-24"
) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
    """
    Strict time-based split:
      - Train: start to 2018-12-31
      - Validation: 2019-01-01 to 2019-12-31
      - Test: 2020-01-01 onwards
      - Pre-lockdown Test: 2020-01-01 to 2020-03-24 (before national lockdown)
      
    Rule 1: Never use random splits or shuffling.
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Continuous daily series/dataframe with DatetimeIndex.
    train_end : str
        End date of training set (inclusive).
    val_end : str
        End date of validation set (inclusive).
    lockdown_start : str
        Last day of pre-lockdown conditions.
        
    Returns
    -------
    tuple
        (train, val, test, test_pre_lockdown)
    """
    train = data.loc[:train_end]
    val = data.loc[pd.to_datetime(train_end) + pd.Timedelta(days=1) : val_end]
    test = data.loc[pd.to_datetime(val_end) + pd.Timedelta(days=1) :]
    test_pre_lockdown = test.loc[:lockdown_start]
    
    return train, val, test, test_pre_lockdown


def calculate_metrics(y_true: Union[pd.Series, np.ndarray, list],
                      y_pred: Union[pd.Series, np.ndarray, list]) -> Dict[str, float]:
    """
    Shared evaluation metrics function across ALL models and cities.
    Computes:
      - MAE (Mean Absolute Error, Primary Metric)
      - RMSE (Root Mean Squared Error)
      - MAPE (Mean Absolute Percentage Error, protected against division by zero)
      - Bucket_Acc (Percentage of predictions in the same CPCB AQI bucket)
      
    Parameters
    ----------
    y_true : array-like
        Ground truth AQI values.
    y_pred : array-like
        Predicted AQI values.
        
    Returns
    -------
    dict
        {'MAE': float, 'RMSE': float, 'MAPE': float, 'Bucket_Acc': float}
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    
    # Filter out any missing values in ground truth or predictions
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_mask):
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "Bucket_Acc": np.nan}
        
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    # Clip denominator at 1.0 to prevent division by zero or infinite explosion
    denom = np.clip(np.abs(yt), 1.0, None)
    mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
    
    # AQI bucket accuracy
    true_buckets = to_aqi_bucket(yt)
    pred_buckets = to_aqi_bucket(yp)
    bucket_acc = float(np.mean(true_buckets == pred_buckets) * 100.0)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "Bucket_Acc": round(bucket_acc, 2)
    }


def to_aqi_bucket(values: Union[pd.Series, np.ndarray, float]) -> np.ndarray:
    """
    Maps continuous AQI numbers to standard CPCB qualitative air quality categories.
    """
    vals = np.asarray(values, dtype=float)
    result = np.empty(vals.shape, dtype=object)
    
    for low, high, label in AQI_BUCKETS:
        mask = (vals >= low) & (vals <= high)
        result[mask] = label
        
    # Catch any negative or extreme numbers
    result[vals < 0] = "Good"
    result[result == None] = "Severe"  # noqa: E711
    return result


def plot_forecast(
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    pred: pd.Series,
    city: str,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Standard visualization helper for model predictions vs actuals.
    
    Highlights:
      - Actual history (Train, Validation, Test)
      - Model forecast line
      - Cutoff dates (Train/Val, Val/Test)
      - Lockdown onset vertical line (2020-03-25)
      - Annotated MAE/RMSE/MAPE box
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot historical data
    ax.plot(train.index, train.values, label="Train (2015-2018)", color="#1f77b4", alpha=0.5, lw=1.2)
    ax.plot(val.index, val.values, label="Val (2019)", color="#2ca02c", alpha=0.7, lw=1.2)
    ax.plot(test.index, test.values, label="Test Actual (2020)", color="#333333", alpha=0.9, lw=1.5)
    
    # Plot predicted values
    pred_series = pd.Series(pred, index=test.index[:len(pred)])
    ax.plot(pred_series.index, pred_series.values, label=f"Forecast ({model_name})", color="#d62728", lw=1.8, linestyle="--")
    
    # Boundary and event markers
    ax.axvline(pd.to_datetime("2019-01-01"), color="gray", linestyle=":", lw=1.2, label="Train/Val Split")
    ax.axvline(pd.to_datetime("2020-01-01"), color="black", linestyle=":", lw=1.2, label="Val/Test Split")
    ax.axvline(pd.to_datetime("2020-03-25"), color="darkorange", linestyle="-.", lw=1.5, label="Lockdown (25-Mar-2020)")
    
    # Metrics box
    if metrics:
        metric_txt = (
            f"Test Metrics:\n"
            f"MAE: {metrics.get('MAE', np.nan):.2f}\n"
            f"RMSE: {metrics.get('RMSE', np.nan):.2f}\n"
            f"MAPE: {metrics.get('MAPE', np.nan):.2f}%\n"
            f"Bucket Acc: {metrics.get('Bucket_Acc', np.nan):.1f}%"
        )
        ax.text(
            0.02, 0.95, metric_txt,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
            fontsize=10
        )
        
    ax.set_title(f"{city} Daily AQI - {model_name} Forecast vs Actuals", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("AQI", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        
    return fig
