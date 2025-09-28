import pandas as pd
import yaml
import os

def load_tariff_yaml(rate_code):
    yaml_path = os.path.join(os.path.dirname(__file__), 'tariffs.yaml')
    with open(yaml_path, 'r') as f:
        tariffs = yaml.safe_load(f)
    return tariffs[rate_code]

def build_tariff_timeseries(rate_code: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Given a rate code and a DatetimeIndex, return a DataFrame with columns:
    - energy_import_rate_kwh
    - energy_export_rate_kwh
    - demand_charge_kw
    """
    tariff = load_tariff_yaml(rate_code)
    # Prepare output
    df = pd.DataFrame(index=index)
    df['energy_import_rate_kwh'] = 0.0
    df['energy_export_rate_kwh'] = 0.0
    df['demand_charge_kw'] = 0.0

    for season in tariff['seasons']:
        # Determine which months are in this season
        months = season['period']
        # Mask for this season
        season_mask = df.index.month.isin(months)
        for period in season['periods']:
            hours = period['hours']
            period_name = period['name']
            # Mask for this period
            period_mask = df.index.hour.isin(hours)
            mask = season_mask & period_mask
            # Set rates for this period
            import_rates = season['rates']['energy_import_rate_kwh']
            export_rates = season['rates']['energy_export_rate_kwh']
            demand_rates = season['rates']['demand_charge_kw']
            # Set import rate
            if period_name in import_rates:
                df.loc[mask, 'energy_import_rate_kwh'] = import_rates[period_name]
            elif 'off-peak' in import_rates and period_name == 'off-peak':
                df.loc[mask, 'energy_import_rate_kwh'] = import_rates['off-peak']
            # Set export rate (default or by period)
            if export_rates.get(period_name) is not None:
                df.loc[mask, 'energy_export_rate_kwh'] = export_rates[period_name]
            elif export_rates.get('default') is not None:
                df.loc[mask, 'energy_export_rate_kwh'] = export_rates['default']
            # Set demand charge (by period or 'any')
            if period_name in demand_rates:
                df.loc[mask, 'demand_charge_kw'] = demand_rates[period_name]
            elif 'any' in demand_rates:
                df.loc[mask, 'demand_charge_kw'] = demand_rates['any']
    return df

