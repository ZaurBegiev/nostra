"""
Interface to Exponential Smoothing model implementation from statsmodels
Reference: https://www.statsmodels.org/devel/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
"""

import logging
import warnings
from typing import Tuple, Union

from hyperopt import hp
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecasting.dataset import BaseWorker, Dataset
from forecasting.metrics import mae

warnings.filterwarnings('ignore')
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)


class Worker(BaseWorker):
    """Local worker class for storing current tuning state and objective function"""
    def __init__(self, data: Dataset, max_evals: int = 70):
        super().__init__(data)

        self.constructor_params = ('trend', 'damped', 'seasonal')
        self.constructor_params_amount = len(self.constructor_params)
        self.fit_params = ('smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'use_boxcox')

        self.space = [
            hp.choice('trend', ['additive', 'multiplicative', None]),
            hp.choice('damped', [True, False]),
            hp.choice('seasonal', ['additive', 'multiplicative', None]),
            hp.uniform('smoothing_slope', .01, .5),
            hp.uniform('smoothing_trend', .01, .5),
            hp.uniform('smoothing_seasonal', .01, .5),
            hp.choice('use_boxcox', [True, False, 'log'])
        ]

        self.max_evals = max_evals

    def objective(self, params: Tuple[Union[int, float]]) -> float:
        """Objective function to minimize during parameter optimization"""
        m = self.constructor_params_amount
        to_constructor = dict(zip(self.constructor_params, params[0:m]))
        to_fit = dict(zip(self.fit_params, params[m:]))

        try:
            model = ExponentialSmoothing(
                endog=self.train,
                **to_constructor,
                seasonal_periods=self.seasonal_periods,
            )
            model_fit = model.fit(**to_fit)
            prediction = model_fit.forecast(len(self.test))
            forecast = model_fit.forecast(self.forecast_and_test_size)
        except ValueError:
            try:
                model = ExponentialSmoothing(
                    endog=self.train)
                model_fit = model.fit()
                prediction = model_fit.forecast(len(self.test))
                forecast = model_fit.forecast(self.forecast_and_test_size)
            except Exception:
                return np.inf

        if sum(np.isnan(prediction)) > 0:
            return np.inf

        loss = mae(self.test, prediction)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_param_values = params
            if not self.is_on_cv:
                self.best_prediction = forecast
        return loss
