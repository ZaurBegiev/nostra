"""Loss functions"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


def mae(y_actual: pd.Series, y_prediction: pd.Series) -> float:
    """Get Mean Absolute Error"""
    # return mean_absolute_error(y_actual.values, y_prediction.values)
    return mean_absolute_error(y_actual, y_prediction)


def mape(y_actual: pd.Series, y_prediction: pd.Series) -> float:
    """Get Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_actual - y_prediction))/ y_actual)


