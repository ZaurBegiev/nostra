"""Classes and static functions operating data preparation and training process"""

from typing import Dict, Union, List, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, adfuller

from forecasting import csv_parser
from forecasting.config import TRAIN_SPLIT_SIZE, HORIZONS


class Dataset:
    """
    Class holding current data and its precalculated properties.

    Attributes:
        n:
            Int, number of input datapoints
        freq:
            String defined in user input. Pandas frequency string, designating a period between datapoints.
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        stationarity:
            Boolean value of series stationarity derived from Augmented Dickey-Fuller test.
        is_seasonal:
            Boolean value of series seasonality derived from autocorellations.
        seasonality_period:
            Number of periods of detected seasonality
        X_train, y_train, X_test, y_test:
            Train-test split ranges
        test_start_index:
            The value of first X_test index in ascending order
        cv_splits:
            Preamptively splitted cross-validation folds for further CV checks
        df:
            DataFrame containing parsed input data. Columns are named 'X' and 'y'.
        df_:
            Dataframe with additional features, used in regression models
        X_train_, X_test_:
            Split train-test data with additional features
        X_train_scaled, X_test_scaled:
            Standardized train-test splits of data with additional features
        cv_splits_:
            Preamptively splitted cross-validation folds of data with additional features
            for further CV checks
        standard_scaler:
            Stored instance of a Standard scaler that standardized train-test data for
            regression modelling
        forecast_length:
            Defined in user input. Amount of datapoints required to predict
        future_datetimes:
            A daterange of datetimes to predict
        future_df:
            DataFrame with additional features added to future datepoints
    """
    def __init__(self, user_input):
        """
        Args:
            user_input:
                    file_path: String containing a path to the file
                    date_order: String containing an order of date parts
                        (e.g. 'DMY', 'MYD'...)
                    freq: Pandas frequency string.
                    forecast_length: Number of datapoints to predict
        """
        df = csv_parser.run(user_input)

        df.columns = ['X', 'y']
        df.y = df.y.astype('float')

        df_clean = clean_from_outliers(df)

        self.n = len(df)
        self.freq = pd.infer_freq(df.X)
        if self.freq is None:
            self.freq = 'D'

        self.stationarity = check_stationarity(df_clean.y)
        self.is_seasonal, self.seasonality_period = check_seasonality(df_clean.y)

        self.X_train, self.y_train, self.X_test, self.y_test,\
            self.test_start_index = timeseries_train_test_split(df_clean, self.n).values()
        self.test_size = len(self.y_test)
        self.cv_splits = get_cv_splits(df_clean, self.n)

        self.df = df

        df_ = add_features(df_clean, self.seasonality_period)
        add_code_mean(df_, df_)

        self.df_ = df_
        self.first_timestamp = df.X[0]

        self.X_train_, self.X_test_ = panel_data_train_test_split(df_, self.n).values()
        standard_scaler = StandardScaler()
        self.X_train_scaled = standard_scaler.fit_transform(fill_nan(self.X_train_).values)
        self.X_test_scaled = standard_scaler.transform(fill_nan(self.X_test_).values)

        self.cv_splits_ = self._cv_split_panel_data()
        self.standard_scaler = standard_scaler

        self.forecast_length = user_input['forecast_length']

        self.future_datetimes = self._get_future_datetimes()
        future_df = get_future_dataframe(self.future_datetimes)
        future_df = fill_nan(add_features(future_df, self.seasonality_period))
        add_code_mean(future_df, df_)
        scale_future(self.standard_scaler, future_df)
        self.future_df = future_df
        self.df['X'] = pd.to_datetime(self.df['X'])

    def __repr__(self):
        header = f'<Dataset>({self.n} datapoints)'
        df_head = f'Head:\n{str(self.df[["X", "y"]].head())}'
        frequency = f'Frequency: {self.freq}'
        seasonality = f'Seasonality: {self.is_seasonal, self.seasonality_period}'
        stationarity = f'Stationarity: {self.stationarity}'
        output = f'{header}\n{df_head}\n{frequency}\n{seasonality}\n{stationarity}\n'
        return output

    def _cv_split_panel_data(self) -> List[Dict[str, Union[pd.DataFrame, pd.Series]]]:
        cv_splits = []
        df_ = fill_nan(self.df_)
        for _, split_values in self.cv_splits.items():
            train_indices = split_values['y_train'].index
            test_indices = split_values['y_test'].index
            train = df_.iloc[train_indices]
            test = df_.iloc[test_indices]
            X_cols = df_.columns.drop('y')
            cv_splits.append(
                {
                    'X_train': train[X_cols],
                    'y_train': train['y'],
                    'X_test': test[X_cols],
                    'y_test': test['y'],
                }
            )
        return cv_splits

    def _get_future_datetimes(self) -> pd.date_range:
        return pd.date_range(
            self.df.X[len(self.df.X) - 1],
            periods=self.forecast_length + 1,
            freq=self.freq,
            closed='right'
        )


class BaseWorker:
    """
    Auxiliary class for storing current tuning state

    Attributes:
        train:
            Current train set
        test:
            Current test set
        seasonal_periods:
            Link to a dataset seasonal periods
        forecast_and_test_size:
            Sum length of test and prediction sizes
        best_prediction:
            Prediction(ndarray) with best mape during current tuning session
        is_on_cv:
         Boolean flag of worker being on cross-validation check at the moment
    """
    def __init__(self, data: Dataset):
        self.train = data.y_train
        self.test = data.y_test
        self.seasonal_periods = data.seasonality_period
        self.forecast_and_test_size = data.forecast_length + len(data.y_test)

        self.best_prediction = None
        self.best_param_values = None
        self.best_loss = np.inf

        self.is_on_cv = False

    def get_cv_score(self, data: Dataset, objective: Callable) -> float:
        self.is_on_cv = True
        for _, split_values in data.cv_splits.items():
            mapes = []
            self.train = split_values['y_train']
            self.test = split_values['y_test']
            mape_ = objective(self.best_param_values)
            mapes.append(mape_)
        self.is_on_cv = False
        return np.mean(mapes)


def _bartlett_formula(r_values: np.ndarray, m: int, length: int) -> float:
    if m == 1:
        return np.math.sqrt(1 / length)
    else:
        return np.math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r_values[:m - 1]))) / length)


def _check_outlier(value: float, mean: float, std: float) -> bool:
    return (value > mean + 3 * std) or (value < mean - 3 * std)


def _code_mean(data: pd.DataFrame, cat_feature: str, real_feature: str) -> Dict[int, float]:
    return dict(data.groupby(cat_feature)[real_feature].mean())


def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].fillna(np.median(df[col].dropna()))
    return df


def check_seasonality(y: pd.Series) -> Tuple[bool, int]:
    max_lag, alpha = 24, .05
    n_unique = np.unique(y).shape[0]

    if n_unique == 1:
        return False, 0

    r = acf(y, nlags=max_lag, fft=False)
    gradient = np.gradient(r)
    gradient_signs_changes = np.diff(np.sign(gradient))

    if len(np.nonzero((gradient_signs_changes == -2))[0]) == 0:
        return False, 0

    candidates = np.nonzero((gradient_signs_changes == -2))[0].tolist()
    candidates = [i if r[i] >= r[i + 1] else i + 1 for i in candidates]

    r = r[1:]

    band_upper = r.mean() + norm.ppf(1 - alpha / 2) * r.var()

    for candidate in candidates:
        stat = _bartlett_formula(r, candidate - 1, len(y))
        if r[candidate - 1] > stat * band_upper:
            return True, candidate
    return False, 0


def add_code_mean(df: pd.DataFrame, df_from: pd.DataFrame) -> None:
    cat_features = ['hour', 'weekday', 'is_weekend']

    for feature in cat_features:
        df[f'average_{feature}'] = df[feature].apply(lambda value: _code_mean(df_from, feature, 'y')[value])


def add_features(df: pd.DataFrame, seasonality_period: int) -> pd.DataFrame:
    df['X'] = df['X'].apply(lambda timestamp: timestamp.value)
    for lag in range(seasonality_period - 1, 25):
        df[f'lag_{lag}'] = df[df.columns[0]].shift(lag)

    time_index = pd.to_datetime(df['X'])

    df["hour"] = time_index.dt.hour
    df["weekday"] = time_index.dt.weekday
    df['is_weekend'] = df.weekday.isin([5, 6]) * 1
    return df


def check_stationarity(y: pd.Series) -> bool:
    results = adfuller(y, autolag='AIC')
    return results[0] < results[4]['5%']


def clean_from_outliers(df: pd.DataFrame) -> pd.DataFrame:
    std = np.std(df.y)
    mean = np.mean(df.y)

    df['is_anomaly'] = df.y.apply(lambda value: _check_outlier(value, mean, std))
    df_for_interpolation = df.set_index(pd.to_datetime(df['X']))
    df_for_interpolation.loc[df_for_interpolation['is_anomaly'], 'y'] = np.nan
    df_for_interpolation.drop(['X', 'is_anomaly'], axis=1, inplace=True)
    df.drop('is_anomaly', axis=1, inplace=True)

    df_interpolated = df_for_interpolation.interpolate(method='time').reset_index()

    if df_interpolated['y'].isna().sum() > 0:
        return df
    else:
        return df_interpolated


def get_cv_splits(df: pd.DataFrame, n: int) -> Dict[float, Dict[str, pd.Series]]:
    X, y, n = df.X, df.y, n
    tscv = TimeSeriesSplit(n_splits=5)
    cv_splits = {}
    for horizon in HORIZONS:
        max_length = int(n * horizon)
        X_cut = X.iloc[:max_length]
        y_cut = y.iloc[:max_length]
        for train_indices, test_indices in tscv.split(X_cut):
            split = {
                'X_train': X_cut.iloc[train_indices],
                'y_train': y_cut.iloc[train_indices],
                'X_test': X_cut.iloc[test_indices],
                'y_test': y_cut.iloc[test_indices],
                'test_start_index': test_indices[0],
            }
            cv_splits[horizon] = split
    return cv_splits


def get_future_dataframe(datetimes: pd.date_range) -> pd.DataFrame:
    future = pd.DataFrame(datetimes)
    future.columns = ['X']
    return future


def panel_data_train_test_split(df: pd.DataFrame, n: int) -> Dict[str, pd.Series]:
    X, y, n = df[df.columns.drop('y')], df.y, n
    train_size = int(n * TRAIN_SPLIT_SIZE)
    return {
        'X_train_': X.iloc[:train_size],
        'X_test_': X.iloc[train_size:],
        }


def scale_future(scaler: StandardScaler, future: pd.DataFrame) -> pd.DataFrame:
    cols = future.columns[:]
    scaled_values = scaler.transform(future.values)
    future = pd.DataFrame(scaled_values)
    future.columns = cols
    return future


def stack_test_and_future(test: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    test = pd.DataFrame(test)
    test.columns = future.columns
    return pd.concat([test, future]).values


def timeseries_train_test_split(df: pd.DataFrame, n: int) -> Dict[str, pd.Series]:
    X, y, n = df.X, df.y, n
    train_size = int(n * TRAIN_SPLIT_SIZE)
    _ = X.iloc[:train_size]
    return {
        'X_train': X.iloc[:train_size],
        'y_train': y.iloc[:train_size],
        'X_test': X.iloc[train_size:],
        'y_test': y.iloc[train_size:],
        'test_start_index': train_size,
    }
