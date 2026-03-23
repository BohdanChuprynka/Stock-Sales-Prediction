import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.feature_selection import mutual_info_regression


def evaluate_model(y_pred, y_true):
    """Compute regression metrics for model evaluation.

    Returns dict with MAE, MSE, RMSE, MAX_ERROR, RMSLE, R2.
    Uses RMSLE instead of MAPE (MAPE is unreliable with zero-valued targets).
    """
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": metrics.mean_squared_error(y_true, y_pred),
        "RMSE": metrics.mean_squared_error(y_true, y_pred, squared=False),
        "MAX_ERROR": metrics.max_error(y_true, y_pred),
        "RMSLE": np.sqrt(
            np.mean(
                (np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0))) ** 2
            )
        ),
        "R2": metrics.r2_score(y_true, y_pred),
    }


def feature_importance(X_train, y_train, feature_names, limit_rows=10000, top_features=10):
    """Compute mutual information regression for feature importance.

    Args:
        X_train: Training features array
        y_train: Training target array
        feature_names: List of feature names
        limit_rows: Max rows to use (for computational efficiency)
        top_features: Number of top features to return
    """
    mutual_info = mutual_info_regression(X_train[:limit_rows], y_train[:limit_rows])
    mutual_df = pd.DataFrame({
        "Features": pd.Series(feature_names),
        "Mutual_Gain": mutual_info,
    })
    return mutual_df.sort_values(by="Mutual_Gain", ascending=False).head(top_features)
