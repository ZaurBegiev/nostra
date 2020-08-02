"""Main prediction pipeline"""

import logging
import warnings
from typing import Any, Dict, Tuple

from hyperopt import fmin, tpe, hp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random.mtrand import RandomState
import pandas as pd
from sklearn.metrics import r2_score

from forecasting import user_inputs
from forecasting.config import RANDOM_SEED, DEBUG_EVAL_NUM
from forecasting.dataset import Dataset
from forecasting.metrics import mae, mape
from forecasting.models import arima, regressions, theta, exponential_smoothing, fft

warnings.filterwarnings('ignore')
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)


def tune_predict(data: Dataset, worker_class: Any, debug: bool) -> Dict:
    """Optimize parameters of given models on train and test samples"""
    w = worker_class(data, max_evals=DEBUG_EVAL_NUM) if debug else worker_class(data)

    fmin(
        fn=w.objective,
        space=w.space,
        algo=tpe.suggest,
        max_evals=w.max_evals,
        rstate=RandomState(RANDOM_SEED),
    )

    prediction = w.best_prediction if isinstance(w.best_prediction, np.ndarray) else w.best_prediction

    try:
        prediction = prediction.values
    except AttributeError:
        pass

    return {
        'forecast': prediction,
        'score': mae(prediction[:data.test_size], data.y_test),
        'cv_score': w.get_cv_score(data, w.objective)
    }


def get_best_linear_combination(data: Dataset, predictions: Dict, debug: bool = False) -> Dict:
    """Find optimal linear combination of given models to minimize loss"""
    max_evals = 20 if debug else 500
    current_results = {
        'current_best_prediction': None,
        'current_best_loss': np.inf
    }
    test_size = len(data.y_test)
    y_test = data.y_test

    prediction_list = [prediction['forecast'] for prediction in predictions.values()]

    space = [
        hp.uniform('Exponential Smoothing', -2, 2),
        hp.uniform('SARIMA', -2, 2),
        hp.uniform('FFT', -2, 2),
        hp.uniform('Theta', -2, 2),
        hp.uniform('Linear Regression', -2, 2),
        hp.uniform('LassoCV', -2, 2),
        hp.uniform('RidgeCV', -2, 2)
    ]

    def _objective(params: Tuple) -> float:
        sum_ = sum(params)
        normalized_params = [param / sum_ for param in params]
        result = np.zeros(len(prediction_list[0]))
        for prediction, param in zip(prediction_list, normalized_params):
            result += prediction * param
        current_loss = mae(result[:test_size], y_test)
        if current_results['current_best_loss'] > current_loss:
            current_results['current_best_loss'] = current_loss
            current_results['current_best_prediction'] = result
        return current_loss

    best = fmin(
        fn=_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        rstate=RandomState(RANDOM_SEED),
    )

    cv_scores = [prediction['cv_score'] for prediction in predictions.values()]
    sum_best = sum(best.values())
    best_normalized = [value / sum_best for value in best.values()]
    weighted_cv_scores = [score * weight for score, weight in zip(cv_scores, best_normalized)]
    estimated_cv_score = sum(weighted_cv_scores)

    return {
        'forecast': current_results['current_best_prediction'],
        'score': current_results['current_best_loss'],
        'cv_score': estimated_cv_score
    }


def make_forecast(user_input: Dict, debug: bool = False) -> Tuple[Dict, pd.DataFrame]:
    """Make a prediction on given user_input"""
    data = Dataset(user_input)

    predictions = {}

    models_to_tune = {
        'Exponential Smoothing': exponential_smoothing,
        'SARIMA': arima,
        'FFT': fft,
        'Theta': theta,
    }

    for model_name, model_class in models_to_tune.items():
        predictions[model_name] = tune_predict(data, model_class.Worker, debug=debug)

    regressions_predictions = regressions.get_results(data)

    results = {**predictions, **regressions_predictions}
    results['Combined'] = get_best_linear_combination(data, results, debug=debug)

    time_index = pd.date_range(start=data.first_timestamp, periods=data.n+data.forecast_length, freq=data.freq)
    df = pd.DataFrame(index=time_index)

    df['Actual'] = np.nan
    df['Actual'].iloc[0:data.n] = data.df.y.values

    columns = list(results.keys())
    for column in columns:
        df[column] = np.nan
        df[column].iloc[data.test_start_index:] = results[column]['forecast']

    results = {
        model_name: {
            'score': values['score'],
            'cv_score': values['cv_score']
        }
        for model_name, values in results.items()
    }

    return results, df


def _choose_best_model(results: Dict) -> str:
    """From the results dict get the name of the model with the lowest loss"""
    result_sums = {model_name: result['score'] + result['cv_score'] for model_name, result in results.items()}
    return sorted(result_sums.items(), key=lambda item: item[1])[0][0]


def get_additional_metrics(df):
    df = df[['Actual', 'Combined']].dropna()
    actual = df['Actual'].values
    pred = df['Combined'].values
    if mape(actual, pred) == np.inf:
        return {
            'r2': r2_score(actual, pred),
            }
    else:
        return {
            'r2': r2_score(actual, pred),
            'mape': mape(actual, pred)
        }


def plot_results(results: Dict, df: pd.DataFrame, only_best: bool = False):
    """Plot the predictions and actual."""
    best_model = _choose_best_model(results)
    labels = dict(zip(df.columns, df.columns))
    for column in df.columns:
        if column != 'Actual':
            formatted_score = '{:.2f}'.format(results[column]['score'])
            formatted_cv_score = '{:.2f}'.format(results[column]['cv_score'])
            labels[column] = f'{labels[column]} {formatted_score}|{formatted_cv_score}'
        if only_best:
            if column in ('Actual', best_model):
                plt.plot(df[column], label=labels[column]+str(get_additional_metrics(df)))
        else:
            plt.plot(df[column], label=labels[column])
    plt.legend()


for user_input in user_inputs.inputs[0:1]:
    results_, df_ = make_forecast(user_input=user_input, debug=True)
    plot_results(results_, df_, only_best=True)

plt.show()
