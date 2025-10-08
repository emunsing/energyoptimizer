import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import itertools as it
from energyoptimizer.batteryopt_utils import shift_copy_dataset_to_new_index


TZ='US/Pacific'

def analytic_input_data(start_dt, end_dt, freq='H', tz=TZ, anchor='left'):
    """
    end_dt is *inclusive*

    expected failing input cases:
    - Less-than-full-year (2015-01-01 to 2015-12-31 00:00; raise error);
    - 1-year hourly input data but missing hours
    - timezone unaware input_dataset
    - timezone unaware new_time_index

    Input Cases:
    - 15-minute vs. 30-min vs. 1-hr input
    - Leap-year vs non-leap-year input
    - Input span: 1-year on-year-end (jan 1 - Dec 31), 1-year off-year-end (e.g. Jun 1- May 31), multi-year on-year-end (2015-01-01:2016-12-31), multi-yaer off-year-end (2015-06-01:2017-05-31)


    For Input data:

    populate data so that we're able to determine where a datapoint came from:
    - year_number (0, 1, 2, ...) * 100
    - month (1-12) * 1
    - hour_of_day (0-23) * 0.01

    So, for example, Jan 1, 00:00 of year 0 = 000 + 1 + 0 = 1.00
    Jun 15, 12:00 of year 2 = 200 + 6 + 12*0.01 = 206.12
    """
    data_series = pd.Series(0.0,
                            index=pd.date_range(start=start_dt, end=end_dt, freq=freq, tz=tz),
                            name='test_data'
                            )

    dt = relativedelta(end_dt, start_dt)
    fractional_n_years_in_dataset = dt.years + dt.days / 365.0 + dt.hours / 8760.0 + dt.minutes / 525600.0
    n_years_in_dataset = int(np.floor(fractional_n_years_in_dataset))
    if anchor == 'left':
        years_start_at = start_dt
    elif anchor == 'right':
        years_start_at = end_dt - relativedelta(years=n_years_in_dataset)
    else:
        raise ValueError("anchor must be either 'left' or 'right'")

    for y in range(n_years_in_dataset):
        yr_start = years_start_at + relativedelta(years=y)
        yr_end = years_start_at + relativedelta(years=y+1)
        data_series.loc[yr_start:yr_end] += y * 100

    data_series += data_series.index.month * 1
    data_series += data_series.index.hour * 0.01

    return data_series


# Build cartesian product of test cases we want:

input_spans = {'1yr_simple': ('2021-01-01', '2021-12-31 23:59'),
               '1yr_offcenter': ('2021-06-01', '2022-05-31 23:59'),  # 1-year on-year-end, 1-year off-year-end]
               '2yr_simple': ('2020-01-01', '2021-12-31 23:59'),
                '2yr_offcenter': ('2020-06-01', '2022-05-31 23:59')  # multi-year on-year-end, multi-year off-year-end
               }
freqs = ['15min', '30min', '1h']

input_cases = [
    (label, start_dt, end_dt, freq)
    for (label, (start_dt, end_dt)), freq in it.product(input_spans.items(), freqs)
]

@pytest.fixture(params=input_cases, ids=lambda x: f"input_{x[0]}-{x[3]}")
def sample_input_data(request, tz=TZ):
    label, start_dt, end_dt, freq = request.param
    return analytic_input_data(pd.Timestamp(start_dt, tz=tz), pd.Timestamp(end_dt, tz=tz), freq=freq)


target_index_cases = {
        'single_year': pd.date_range(
            start='2025-01-01 00:00:00',
            end='2025-12-31 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
        'leap_year': pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-12-31 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
        'partial_year': pd.date_range(
            start='2025-06-01 00:00:00',
            end='2025-11-30 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
        'year_and_a_half': pd.date_range(
            start='2025-06-01 00:00:00',
            end='2026-12-31 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
        'multi_year': pd.date_range(
            start='2026-01-01 00:00:00',
            end='2028-12-31 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
        '10_year': pd.date_range(
            start='2025-01-01 00:00:00',
            end='2034-12-31 23:30:00',
            freq='30T',
            tz='US/Pacific'
        ),
    }

@pytest.fixture(params=list(target_index_cases.items()), ids=lambda x: f"target_{x[0]}")
def target_index(request):
    _, idx = request.param
    return idx

def test_shift_copy_dataset_to_new_index(sample_input_data, target_index):
    """Test the main function with a variety of input and output cases"""
    output_data = shift_copy_dataset_to_new_index(sample_input_data, target_index, shuffle_years=False)

    assert output_data.index.equals(target_index), "Output index does not match target index"
    assert not output_data.isnull().any(), "Output data contains null values"


def test_temp_shift_copy_dataset_to_new_index():
    # start_dt, end_dt = input_spans['1yr_offcenter']
    start_dt, end_dt = input_spans['2yr_simple']
    freq = '30T'
    target_index = target_index_cases['leap_year']

    sample_input_data = analytic_input_data(pd.Timestamp(start_dt, tz=TZ), pd.Timestamp(end_dt, tz=TZ), freq=freq)

    """Test the main function with a variety of input and output cases"""
    output_data = shift_copy_dataset_to_new_index(sample_input_data, target_index, shuffle_years=False)

    assert output_data.index.equals(target_index), "Output index does not match target index"
    assert not output_data.isnull().any(), "Output data contains null values"


#
# def test_error_cases(analytic_input_data):
#     """Test expected error cases - these should work even with current bugs"""
#     # Test with less than full year input
#     short_data = analytic_input_data.iloc[:1000]  # Less than a year
#
#     with pytest.raises(AssertionError, match="Input dataset must span more than one year"):
#         shift_copy_dataset_to_new_index(
#             short_data,
#             pd.date_range('2025-01-01', '2025-12-31', freq='h', tz='US/Pacific')
#         )
#
#     # Test with timezone-unaware data
#     tz_unaware_data = analytic_input_data.copy()
#     tz_unaware_data.index = tz_unaware_data.index.tz_localize(None)
#
#     with pytest.raises(AssertionError, match="Input dataset must have a fixed frequency"):
#         shift_copy_dataset_to_new_index(
#             tz_unaware_data,
#             pd.date_range('2025-01-01', '2025-12-31', freq='h', tz='US/Pacific')
#         )