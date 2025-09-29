import pandas as pd


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
