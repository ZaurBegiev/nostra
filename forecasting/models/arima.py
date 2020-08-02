"""
Interface to ARIMA model implementation from statsmodels
Reference: https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html
"""

import logging
import warnings
from typing import Tuple, Union

from hyperopt import hp
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from forecasting.dataset import BaseWorker, Dataset
from forecasting.metrics import mae

warnings.filterwarnings('ignore')
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)


class Worker(BaseWorker):
    """Local worker class for storing current tuning state and objective function"""
    def __init__(self, data: Dataset, max_evals: int = 100):
        super().__init__(data)

        self.constructor_params = ('p', 'd', 'q', 'P', 'D', 'Q', 's', 'trend')
        self.constructor_params_amount = len(self.constructor_params)

        r = list(range(4))
        self.space = [
            hp.choice('p', r),
            hp.choice('d', r),
            hp.choice('q', r),
            hp.choice('P', r),
            hp.choice('D', r),
            hp.choice('Q', r),
            hp.choice('s', r),
            hp.choice('trend', ['n', 'c', 't', 'ct'])
        ]

        self.max_evals = max_evals

    def objective(self, params: Tuple[Union[int, float]]) -> float:
        """Objective function to minimize during parameter optimization"""
        m = self.constructor_params_amount
        params_dict = dict(zip(self.constructor_params, params[0:m]))

        try:
            model = ARIMA(
                endog=self.train,
                order=(
                    params_dict['p'],
                    params_dict['d'],
                    params_dict['q'],
                ),
                seasonal_order=(
                    params_dict['P'],
                    params_dict['D'],
                    params_dict['Q'],
                    params_dict['s'],
                ),
                trend=params_dict['trend']
            )

            model_fit = model.fit()
            prediction = model_fit.forecast(len(self.test))
            forecast = model_fit.forecast(self.forecast_and_test_size)

        except (ValueError, np.linalg.LinAlgError):
            try:
                model = ARIMA(endog=self.train)
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
