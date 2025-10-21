"""
Metrics calculation utilities
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive forecast metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Symmetric MAPE (handles zero values better)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Accuracy (1 - normalized MAE)
    accuracy = (1 - mae / np.mean(y_true)) * 100
    
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "accuracy": float(accuracy)
    }


def calculate_confidence_interval(predictions: np.ndarray, 
                                  confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions
    
    Args:
        predictions: Array of predictions
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    std = np.std(predictions)
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    
    margin = z_score * std
    lower_bound = predictions - margin
    upper_bound = predictions + margin
    
    return lower_bound, upper_bound
