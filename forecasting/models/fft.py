"""
Interface to Fast Fourier Transform model implementation from darts
Source: https://github.com/unit8co/darts/blob/master/darts/models/fft.py
"""

import logging
import warnings
from typing import Tuple, Union

from darts import TimeSeries
from darts.models import FFT
from hyperopt import hp
import numpy as np

from forecasting.dataset import BaseWorker, Dataset
from forecasting.metrics import mae

warnings.filterwarnings('ignore')
logging.getLogger("hyperopt.tpe").setLevel(logging.ERROR)
logging.getLogger("darts.models.theta").setLevel(logging.ERROR)


class Worker(BaseWorker):
    """Local worker class for storing current tuning state and objective function"""
    def __init__(self, data: Dataset, max_evals: int = 80):
        super().__init__(data)

        self.constructor_params = ('nr_freqs_to_keep', 'trend', 'trend_poly_degree')
        self.constructor_params_amount = len(self.constructor_params)

        self.space = [
            hp.choice('nr_freqs_to_keep', list(range(5, 15))),
            hp.choice('trend', ['poly', 'exp']),
            hp.choice('trend_poly_degree', list(range(4))),
        ]
        self.max_evals = max_evals
        self.df = data.df
        self.y_train = data.df

    def objective(self, params: Tuple[Union[int, float]]) -> float:
        """Objective function to minimize during parameter optimization"""
        m = self.constructor_params_amount
        to_constructor = dict(zip(self.constructor_params, params[0:m]))

        try:
            model = FFT(
                **to_constructor,
                required_matches=None,
            )

            ts = TimeSeries.from_dataframe(self.df.iloc[:len(self.y_train)], time_col='X', value_cols='y')
            model.fit(ts)
            prediction = model.predict(len(self.test)).values().reshape(-1)
            forecast = model.predict(self.forecast_and_test_size)
        except ValueError:
            try:
                model = FFT()
                ts = TimeSeries.from_dataframe(self.df.iloc[:len(self.y_train)], time_col='X', value_cols='y')
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
