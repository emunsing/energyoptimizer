import pandas as pd
import numpy as np
import random
from dateutil.relativedelta import relativedelta


def check_datetimeindex_regular(input_dataset: pd.DataFrame | pd.Series | pd.DatetimeIndex):
    if isinstance(input_dataset, (pd.DataFrame, pd.Series)):
        input_dataset = input_dataset.index

    # TODO: Is .freq a strong check that there are no missing timestamps?
    assert isinstance(input_dataset, pd.DatetimeIndex)
    assert input_dataset.freq is not None, "Input dataset must have a fixed frequency"
    assert input_dataset.tz is not None


def create_time_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_df, pd.Series):
        input_df = input_df.to_frame()
    input_df['weekday'] = input_df.index.weekday
    input_df['month'] = input_df.index.month
    input_df['isoweek'] = input_df.index.isocalendar().week
    input_df['hour'] = input_df.index.hour
    input_df['minute'] = input_df.index.minute
    input_df['utcoffset'] = input_df.index.map(lambda x: x.utcoffset())
    return input_df


def shift_copy_dataset_to_new_index(input_dataset: pd.DataFrame | pd.Series,
                                    new_time_index: pd.DatetimeIndex,
                                    shuffle_years: bool = False,
                                    resample_method: str = 'mean',
                                    trim_anchor='right'):

    """
    Shift and copy a dataset to a new time index, preserving autoregressive integrity as much as possible (with exception of the boundaries between years and filling NaNs)
    This will be primarily used for extending historical data to model future performance (e.g. extending 2022 electricity meter data to model 2025-2035 performance)

    NOTE: This is a hard problem! Ideally would need a calendar of holidays mapped to names, so that
    e.g. Christmas is always appropriately observed with an energy usage spike.

    Currently we merge on ['year_number', 'isoweek', 'weekday', 'hour', 'minute', 'utcoffset'] but this means that the
    week numbers can shift as the year's start- and end-dates shift.  We fill blanks using the average of the
    ['year_number', 'month', 'weekday', 'hour', 'minute'] which means that we may fill with disjoint information.

    A better imputation approach may be to use a timeseries model for imputation.

    Gotchas:
    - Leap-years
    - DST shifts
    - Ensure that multi-year input data is
    - Want to be able to utilize information from all years of the input data
    """

    check_datetimeindex_regular(input_dataset)
    check_datetimeindex_regular(new_time_index)

    ending_time_right_bounded = input_dataset.index[-1] + input_dataset.index.freq
    input_dataset_relativedelta = relativedelta(ending_time_right_bounded, input_dataset.index[0])
    input_dataset_full_years = input_dataset_relativedelta.years
    assert input_dataset_full_years >= 1, "Input dataset must span more than one year"

    working_data = input_dataset.copy()
    working_data = working_data.resample(new_time_index.freq).agg(resample_method)
    working_data = working_data.ffill(limit=1)
    ending_time_right_bounded = working_data.index[-1] + working_data.index.freq

    # Trim the input_dataset to a multiple of a year, either anchoring on the right or the left based on trim_anchor
    if trim_anchor == 'right':
        working_data = working_data[ending_time_right_bounded - relativedelta(years=input_dataset_full_years): ]
    elif trim_anchor == 'left':
        working_data = working_data[:input_dataset.index[0] + relativedelta(years=input_dataset_full_years)]
    else:
        raise ValueError("trim_anchor must be either 'right' or 'left'")

    if isinstance(working_data, pd.Series):
        working_data = working_data.to_frame()
    input_columns = working_data.columns

    # For input_dataset: Create a new column 'year_number'.  For each year in the input dataset, assign a sequential number starting from 0.
    working_data['year_number'] = 0
    for yr in range(0, input_dataset_full_years):
        yr_start = working_data.index[0] + relativedelta(years=yr)
        yr_end = working_data.index[0] + relativedelta(years=yr+1)
        working_data.loc[yr_start:yr_end, 'year_number'] = yr

    # For input_dataset: Create new columns isoweekday, isoweek, hour, minute
    working_data = create_time_columns(working_data)


    # For new_time_index: Create new columns isoweekday, isoweek, hour, minute
    # for new_time_index: create new column 'year_number'.  Increment from 0 to n_years_in_input, then either shuffle from the input years or repeat the loop of input years, depending on shuffle_years
    target_index_df = pd.DataFrame(index=new_time_index)
    target_index_df = create_time_columns(target_index_df)

    new_time_index_closed_right_bounded = new_time_index[-1] + new_time_index.freq
    new_time_index_relativedelta = relativedelta(new_time_index_closed_right_bounded, new_time_index[0])
    fractional_n_years_in_target = new_time_index_relativedelta.years + new_time_index_relativedelta.days / 365.0 + new_time_index_relativedelta.hours / 8760.0 + new_time_index_relativedelta.minutes / 525600.0
    ceil_years_in_target = int(np.ceil(fractional_n_years_in_target))
    target_index_df['year_number'] = 0
    for yr in range(0, ceil_years_in_target):
        yr_start = target_index_df.index[0] + relativedelta(years=yr)
        yr_end = target_index_df.index[0] + relativedelta(years=yr+1)

        if yr < input_dataset_full_years:
            reference_year = yr
        else:
            if shuffle_years:
                reference_year = random.choice(range(input_dataset_full_years))
            else:
                reference_year = yr % input_dataset_full_years
        target_index_df.loc[yr_start:yr_end, 'year_number'] = reference_year

    # THE MERGE CRITERIA ARE BESPOKE, AND ARE NOT ANALYTICALLY OPTIMAL
    target_index_df_merged = target_index_df.merge(working_data.drop('month', axis=1), how='left', on=[ 'year_number', 'isoweek', 'weekday', 'hour', 'minute', 'utcoffset'])
    target_index_df_merged.index = target_index_df.index
    target_index_filled = target_index_df_merged.groupby(['year_number', 'month', 'weekday', 'hour', 'minute']).ffill()
    target_index_filled = target_index_filled.ffill(limit=1).bfill(limit=1)
    output_data = target_index_filled[input_columns]

    if output_data.shape[1] == 1:
        output_data = output_data.squeeze()

    return output_data




    # Drop additional columns
    # Assert that there is are no null rows



def merge_solar_and_load_data(elec_usage: pd.Series, solar_ac_estimate: pd.Series, timezone='US/Pacific', verbose=False) -> pd.DataFrame:
    """Because pvlib uses solar data which is in the past, we need to shift the solar data to match
    the actual electricity usage data.
    Right now, this works one year at a time, and relies on a reference set of years hard-coded to be 2019-2021, around the 2020 leap year.
    """
    assert len(elec_usage.index.year.unique()) == 1, "Electricity usage data must be for a single year"
    elec_end_date = elec_usage.index[-1]
    leap_day = None
    for t in elec_usage.index:
        if (t.month==2) & (t.day==29):
            leap_day = t
            break

    if solar_ac_estimate.index[0] <= elec_usage.index[0] and (solar_ac_estimate.index[-1] + pd.DateOffset(hours=1)) >= elec_usage.index[-1]:
        shift_by_yrs = 0
    elif leap_day:
        shift_by_yrs = 2020 - leap_day.year
    elif elec_end_date.month == 12 and elec_end_date.day == 31:
        shift_by_yrs = 2019 - elec_end_date.year
    elif elec_end_date.month > 2:
        shift_by_yrs = 2021 - elec_end_date.year
    else:
        shift_by_yrs = 2019 - elec_end_date.year

    if verbose:
        print("solar_ac_estimate")
        print(solar_ac_estimate.head())
        print("elec_usage")
        print(elec_usage.head())

    solar_ac_estimate.index = (solar_ac_estimate.index.tz_convert('UTC') - pd.DateOffset(years=shift_by_yrs)).tz_convert(timezone)
    solar_ac_estimate = solar_ac_estimate.resample('1h', closed='right').last().ffill()  # Deal with any gaps related to shifted DST; thankfully these are in the night

    site_data = pd.DataFrame(elec_usage).join(solar_ac_estimate, how='left').tz_convert(timezone)
    site_data['solar'] = site_data['solar'].interpolate()
    return site_data
