"""
Interface to Theta model implementation from darts
Source: https://github.com/unit8co/darts/blob/master/darts/models/theta.py
"""

import logging
import warnings
from typing import Union, Tuple

from darts import TimeSeries
from darts.models import Theta
from hyperopt import hp
import numpy as np

from forecasting.dataset import BaseWorker, Dataset
from forecasting.metrics import mae


warnings.filterwarnings('ignore')
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)


class Worker(BaseWorker):
    """Local worker class for storing current tuning state and objective function"""
    def __init__(self, data: Dataset, max_evals: int = 20):
        super().__init__(data)

        self.constructor_params = ('theta', 'mode')
        self.constructor_params_amount = len(self.constructor_params)

        self.space = [
            hp.choice('theta', [0, 1, 3, 4, 5]),
            hp.choice('mode', ['additive', 'multiplicative'])
        ]

        self.max_evals = max_evals
        self.df = data.df
        self.y_train = data.df
        self.freq = data.freq

    def objective(self, params: Tuple[Union[int, float]]) -> float:
        """Objective function to minimize during parameter optimization"""
        m = self.constructor_params_amount
        to_constructor = dict(zip(self.constructor_params, params[0:m]))

        try:
            model = Theta(
                **to_constructor,
                seasonality_period=self.seasonal_periods,
            )

            ts = TimeSeries.from_dataframe(
                self.df.iloc[:len(self.y_train)],
                time_col='X', value_cols='y',
                freq=self.freq
            )

            model.fit(ts)

            prediction = model.predict(len(self.test)).values().reshape(-1)
            forecast = model.predict(self.forecast_and_test_size)

        except ValueError:
            try:
                model = Theta()
                ts = TimeSeries.from_dataframe(
                    self.df.iloc[:len(self.y_train)],
                    time_col='X', value_cols='y',
                    freq=self.freq
                )
                model.fit(ts)
                prediction = model.predict(len(self.test)).values().reshape(-1)
                forecast = model.predict(self.forecast_and_test_size)
            except Exception:
                return np.inf

        if sum(np.isnan(prediction)) > 0:
            return np.inf

        loss = mae(self.test, prediction)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_param_values = params
            if not self.is_on_cv:
                self.best_prediction = forecast.values().reshape(-1)
        return loss
