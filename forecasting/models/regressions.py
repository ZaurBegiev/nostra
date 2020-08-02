"""
Interface for sklearn regression models. References:
Linear Regression:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
LassoCV:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV
RidgeCV:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV
"""

from typing import Any, Dict, Union

import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

from forecasting.dataset import stack_test_and_future, scale_future, Dataset
from forecasting.metrics import mae


def get_prediction(data: Dataset, model_class: Any) -> np.ndarray:
    """Get a prediction for a given model and dataset"""
    current_model = model_class()
    current_model.fit(data.X_train_scaled, data.y_train)

    future_df = scale_future(data.standard_scaler, data.future_df)
    test_and_future = stack_test_and_future(data.X_test_scaled, future_df)
    return current_model.predict(test_and_future)


def get_cv_score(data: Dataset, model_class: Any) -> float:
    """Check cross validation score fro given model and dataset"""
    for split_values in data.cv_splits_:
        losses = []
        X_train = split_values['X_train']
        y_train = split_values['y_train']
        X_test = split_values['X_test']
        y_test = split_values['y_test']

        X_train_scaled = data.standard_scaler.transform(X_train.values)
        X_test_scaled = data.standard_scaler.transform(X_test.values)

        current_model = model_class()
        current_model.fit(X_train_scaled, y_train)

        prediction = current_model.predict(X_test_scaled)

        if sum(np.isnan(prediction)) > 0:
            loss = np.inf
        else:
            loss = mae(y_test, prediction)
        losses.append(loss)
    return np.mean(losses)


def get_results(data: Dataset) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """Combine results from all regression models to a dict"""
    models = {
        'Linear Regression': LinearRegression,
        'LassoCV': LassoCV,
        'RidgeCV': RidgeCV,
    }

    results = {}
    for model_name, model_class in models.items():
        prediction = get_prediction(data, model_class)
        results[model_name] = {
            'forecast': prediction,
            'score': mae(prediction[:data.test_size], data.y_test),
            'cv_score':  get_cv_score(data, model_class)
        }
    return results
