"""Parser of csv paths with missing values check and possible interpolation"""

import csv
import logging
from typing import Tuple, Set, List, Dict, Union

import dateparser
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

sniffer = csv.Sniffer()


def _get_csv_sample(file_path: str) -> Tuple[str, str]:
    """Peek into first 5 rows of a file."""
    with open(file_path, 'r') as f:
        sample = ''
        first_line = f.readline()[:-1]
        for row_num in range(5):
            sample += f.readline()
    return first_line, sample


def _get_dialect(sample: str) -> csv.Dialect:
    """Get csv.Dialect object from sample."""
    dialect = sniffer.sniff(sample)
    return dialect


def _read(filepath: str) -> pd.DataFrame:
    """Parse csv with arbitrary delimiters and quotechars."""
    first_line_, sample_ = _get_csv_sample(filepath)
    dialect = _get_dialect(sample_)
    delimiter, quotechar = dialect.delimiter, dialect.quotechar

    df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            quotechar=quotechar
        )
    df = df.iloc[:, :2]
    logger.debug(f'Table parsed.')
    return df


def _parse_dates(df: pd.DataFrame, date_order: str = None) -> pd.DataFrame:
    """Parse dates from dataframe with given date_order."""
    settings = {'DATE_ORDER': date_order} if date_order else None

    df.iloc[:, 0] = df.iloc[:, 0]\
        .apply(lambda value: dateparser.parse(value, settings=settings))
    nones_count = df.iloc[:, 0].isna().sum()

    if nones_count > 3:
        logger.error(
            f'{nones_count} of {len(df)} datetimes could not be recognised. '
            f'Consider changing format or filling missing dates.'
        )
    return df


def _convert_values_to_ranges(timestamps: Set[pd.Timestamp], full_series: pd.Series) -> List[List]:
    """Get list representation of missing time values, in [range_start, range_end] format."""
    timestamps = sorted(timestamps)
    all_ranges = []
    current_range = []

    for index, timestamp in enumerate(timestamps):
        if not current_range:
            current_range.append(timestamp)
        if index != len(timestamps)-1:
            index_in_full_series = np.where(full_series == timestamp)[0]
            next_expected_value = full_series[index_in_full_series+1]

            if timestamps[index+1] != next_expected_value.values[0]:
                if len(current_range) == 1 and current_range[0] != timestamp:
                    current_range.append(timestamp)
                all_ranges.append(current_range)
                current_range = []
        else:
            current_range.append(timestamp)
            all_ranges.append(current_range)
    return all_ranges


def _get_ranges_string_representation(ranges: List[List]) -> str:
    """Get string representation of ranges list"""
    single_values = [range_ for range_ in ranges if len(range_) == 1]
    ranges = [range_ for range_ in ranges if len(range_) > 1]
    result = 'Missing ranges(inclusive):\n'

    threshold = 5

    for range_ in ranges[0:threshold]:
        result += str(range_[0])+' to ' + str(range_[1]) + '\n'

    if len(ranges) > threshold:
        result += 'and more...\n'

    if single_values:
        result += 'Missing single values: '
        for timestamp in single_values[0:threshold]:
            result += str(timestamp[0]) + '\n'

        if len(single_values) > threshold:
            result += 'and more...\n'
    return result


def _check_integrity(datetimes_column: pd.Series, freq: str) -> Dict[str, Union[bool, Set[pd.Timestamp]]]:
    """Check if our datetimes series is intergal, meaning it doesn't have any missing datetimes."""
    results = {
        'is_integral': None,
        'can_be_interpolated': None,
        'missing_dates': None,
    }

    if not any(datetimes_column):
        logger.error('Invalid datetimes to check integrity.')
        return results

    start, end = min(datetimes_column), max(datetimes_column)
    full_series = pd.date_range(start=start, end=end, freq=freq)

    existing_datetime_count = len(datetimes_column)
    expected_datetime_count = len(full_series)
    if expected_datetime_count < existing_datetime_count:
        logger.error('There are duplicate timestamps in data.')

    full_series = pd.Series(full_series).apply(
        lambda timestamp: pd.Timestamp(timestamp, freq=freq)
    )

    expected_datetimes = set(full_series)
    existing_datetimes = set(datetimes_column)

    if existing_datetime_count == expected_datetime_count and expected_datetimes == existing_datetimes:
        logger.debug('Integrity: True')
        results['is_integral'] = True
    else:
        expected_difference = expected_datetimes - existing_datetimes
        expected_values_count = len(expected_difference)

        ranges = _convert_values_to_ranges(expected_difference, full_series=full_series)
        output = f'Integrity: False. Found {expected_values_count} missing datetimes.\n'
        output += _get_ranges_string_representation(ranges)

        logger.debug(output)

        can_be_interpolated = expected_values_count < (0.05 * existing_datetime_count)
        results['is_integral'] = False
        results['can_be_interpolated'] = can_be_interpolated
        results['missing_dates'] = expected_difference
    return results


def _interpolate(df: pd.DataFrame, missing_dates: Set[pd.Timestamp]) -> pd.DataFrame:
    """Interpolate missing datetimes with pd.Series.interpolate"""
    interpolated_time_index = sorted(list(missing_dates) + df['X'].tolist())
    df_interpolated = pd.DataFrame(interpolated_time_index, columns=['X'])

    df_interpolated = df_interpolated.merge(df, how='left', on='X')
    df_interpolated['y'] = df_interpolated['y'].interpolate()
    return df_interpolated


def run(user_input: Dict[str, str]) -> pd.DataFrame:
    """Perform parsing pipeline"""
    file_path = user_input['file_path']

    logger.info(f'Path: {file_path}')

    df = _read(file_path)
    if df is not None:
        try:
            df = _parse_dates(df, date_order=user_input['date_order'])
            logger.debug(f'Dates parsed.')
        except Exception:
            logger.error(f'Error parsing dates.')
            return pd.DataFrame()

        integrity_results = _check_integrity(df.iloc[:, 0], user_input['freq'])

        if not integrity_results['is_integral']:
            logger.debug(f"Can be interpolated: {integrity_results['can_be_interpolated']}")
            try:
                df = _interpolate(df, integrity_results['missing_dates'])
            except Exception:
                logger.error('Error interpolating df. Specify the exception.')

        logger.info('Parsing - successful.')
    else:
        logger.info('Parsing - failed.')
    return df
