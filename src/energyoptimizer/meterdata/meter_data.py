import pandas as pd


def process_pge_meterdata(fname: str) -> pd.Series:
    header_row = None
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES"):
                header_row = i
                break
    if header_row is None:
        raise ValueError("Header row not found!")

    sample_consumption = pd.read_csv(fname, parse_dates={'Datetime': ['DATE', 'START TIME']}, skiprows=header_row)
    sample_consumption['Datetime'] = pd.DatetimeIndex(sample_consumption['Datetime']).tz_localize('US/Pacific', ambiguous='NaT', nonexistent='NaT')
    sample_consumption = sample_consumption[sample_consumption['Datetime'].notnull()]
    sample_consumption = sample_consumption.set_index('Datetime')
    elec_usage = sample_consumption['USAGE (kWh)'].rename('load')

    elec_end_date = elec_usage.index[-1]
    if elec_usage.index[0] < (elec_end_date - pd.DateOffset(years=1)):
        elec_usage = elec_usage.loc[elec_end_date - pd.DateOffset(years=1):elec_end_date]
    return elec_usage
