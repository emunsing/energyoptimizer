import pandas as pd
import pytest
import matplotlib.pyplot as plt
from src.energyoptimizer.tariff.tariff_utils import build_tariff_timeseries

RATE_CODES = ['PGE_B_19_R']

@pytest.mark.parametrize("rate_code", RATE_CODES)
def test_tariff_generator(rate_code):
    index_by_season = {'winter': pd.date_range('2024-01-01', '2024-01-03', freq='15min'),
                        'spring': pd.date_range('2024-04-01', '2024-04-03', freq='15min'),
                        'summer': pd.date_range('2024-07-01', '2024-07-03', freq='15min'),
                        'autumn': pd.date_range('2024-10-01', '2024-10-03', freq='15min')}
    fig, ax = plt.subplots(nrows=len(index_by_season), figsize=(3*len(index_by_season), 8))

    for i, (season, index) in enumerate(index_by_season.items()):
        print(f"\nTesting tariff generation for {season} season")
        tariff_series = build_tariff_timeseries(rate_code, index)
        assert sorted(tariff_series.columns) == sorted(['energy_export_rate_kwh',
                                                        'energy_import_rate_kwh',
                                                        'demand_charge_kw',
                                                        ])
        assert tariff_series.index.equals(index), f"Tariff series length mismatch for {season} season"
        assert not tariff_series['energy_import_rate_kwh'].isnull().any(), f"Tariff series contains NaN values for {season} season"

        tariff_series[['energy_export_rate_kwh', 'energy_import_rate_kwh']].plot(title=season, ax=ax[i])

        print(f"Tariff series for {season} season:")
        diff_idx = tariff_series.loc[tariff_series.diff().abs().sum(1) > 0].index
        print(tariff_series.loc[diff_idx])

    plt.tight_layout()
    plt.show()
    plt.savefig('/Users/eric/Desktop/tariff_test.png')
    print("Done")

