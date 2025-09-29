import pandas as pd
import numpy as np
import pytest
import hashlib
import json
from datetime import date, datetime
from src.energyoptimizer.tariff.tariff_utils import TariffModel

RATE_CODES = ['PGE_B_19_R']

class TestTariffModel:
    """Test suite for the TariffModel class."""

    def test_tariff_version(self):
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', '2024-01-01', '2024-01-31')

        ref_tariff_config_hash = '0de650a1274a5f534bb27efe1fdf1701'  # 9/29/2025
        tariff_config_hash =hashlib.md5(json.dumps(tariff_model.tariff, sort_keys=True).encode()).hexdigest()
        assert tariff_config_hash == ref_tariff_config_hash, "Tariff config has changed, please verify calculations"


    def test_tariff_model_initialization(self):
        """Test TariffModel initialization with basic parameters."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Test that tariff components are built
        assert hasattr(tariff_model, 'tariff_timeseries')
        assert hasattr(tariff_model, 'demand_charge_categorical_dataframe')
        assert hasattr(tariff_model, 'demand_charge_price_map')
        assert hasattr(tariff_model, 'billing_cycles')
        
        # Test tariff timeseries structure
        assert 'energy_import_rate_kwh' in tariff_model.tariff_timeseries.columns
        assert 'energy_export_rate_kwh' in tariff_model.tariff_timeseries.columns
        assert 'demand_charge_rate_kw' in tariff_model.tariff_timeseries.columns
        
        # Test billing cycles
        assert len(tariff_model.billing_cycles) > 0
        assert all(isinstance(cycle, tuple) and len(cycle) == 2 for cycle in tariff_model.billing_cycles)

    def test_demand_charge_categorical_dataframe(self):
        """Test demand charge categorical dataframe structure."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        df = tariff_model.demand_charge_categorical_dataframe
        
        # Should have columns for each billing cycle-period combination
        # For June 2024 (summer), should have: 2024_june_summer_peak, 2024_june_summer_partial-peak, 2024_june_summer_off-peak
        expected_columns = [
            '2024_june_summer_peak', '2024_june_summer_partial-peak', '2024_june_summer_off-peak'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
            assert df[col].dtype == bool, f"Column {col} should be boolean"
        
        # Test that at least some periods are marked as True
        assert df.any().any(), "At least some periods should be marked as True"

    def test_demand_charge_price_map(self):
        """Test demand charge price map structure."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        price_map = tariff_model.demand_charge_price_map
        
        # Should have entries for each billing cycle-period combination
        # For January 2024 (winter), should have: 2024_january_winter_peak, 2024_january_winter_off-peak
        expected_keys = [
            '2024_january_winter_peak', '2024_january_winter_off-peak'
        ]
        
        for key in expected_keys:
            assert key in price_map, f"Missing price map key: {key}"
            assert isinstance(price_map[key], (int, float)), f"Price for {key} should be numeric"
    
    def test_compute_energy_charge_basic(self):
        """Test basic energy charge computation."""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 6, 2)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)

        # Create a simple power series (constant import)
        power_level = 10
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)  # 10 kW constant import
        
        energy_charge = tariff_model.compute_energy_charge(power_series)
        
        # Should be positive (cost)
        assert energy_charge > 0, "Energy charge should be positive for import"
        
        # Should be reasonable magnitude (10 kW * 24 hours * rate)
        # 5 hrs on-peak, 0.46147/kWh
        # 3 hrs partial-peak, 0.27763/kWh
        # 16 hrs off-peak, 0.21716/kWh
        # = power_level * ((5*0.46147) + (3*0.27763) + (16*0.21716)) = 23.0735 + 8.3289 + 34.7456 = 66.148
        assert energy_charge == power_level * 6.6148
    
    def test_compute_energy_charge_export(self):
        """Test energy charge computation with export."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create a power series with export (negative values)
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(-5.0, index=time_index)  # 5 kW constant export
        
        energy_charge = tariff_model.compute_energy_charge(power_series)
        
        # Should be negative (revenue) or zero if no export rate
        assert energy_charge == 0, "Energy charge should be negative or zero for export"

    def test_mixed_import_export(self):
        """Test with mixed import and export power."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)

        # Create power series with both import and export
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(0.0, index=time_index)

        # Import during day, export during night
        day_mask = time_index.hour.isin(range(6, 18))
        power_series.loc[day_mask] = 20.0  # 20 kW import during day
        power_series.loc[~day_mask] = -5.0  # 5 kW export during night

        bill = tariff_model.compute_total_bill(power_series)

        # Should have positive total bill (net import)
        assert bill['total_bill'] > 0
        assert bill['energy_charge'] > 0  # Net import should result in cost
    
    def test_compute_demand_charge_offpeak_only(self):
        """Test basic demand charge computation."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create a power series with peak demand
        power_level = 50
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)  # 50 kW constant import
        power_series[time_index.hour > 13] = 0.0  # No demand during peak hours

        demand_charge = tariff_model.compute_demand_charge(power_series)
        assert demand_charge == power_level * 39.95

    def test_compute_demand_charge_onpeak_only(self):
        """Test basic demand charge computation."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)

        # Create a power series with peak demand
        power_level = 50
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(0, index=time_index)
        # power_series[power_series.between_time('15:01', '20:00').index] = power_level
        power_series[(time_index.hour >= 16) & (time_index.hour <= 20)] = power_level

        demand_charge = tariff_model.compute_demand_charge(power_series)
        assert demand_charge == power_level * (39.95 + 6.97)

    def test_compute_demand_charge_full(self):
        """Test basic demand charge computation."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)

        # Create a power series with peak demand
        power_level = 50
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)  # 50 kW constant import
        demand_charge = tariff_model.compute_demand_charge(power_series)
        assert demand_charge == power_level * (39.95 + 6.97 + 2.01)

    def test_compute_total_bill(self):
        """Test total bill computation."""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 6, 2)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create a simple power series
        power_level = 10
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)
        
        bill = tariff_model.compute_total_bill(power_series)
        
        # Should have required fields
        assert 'demand_charge' in bill
        assert 'energy_charge' in bill
        assert 'total_bill' in bill
        
        # Total should equal sum of components
        assert abs(bill['total_bill'] - (bill['demand_charge'] + bill['energy_charge'])) < 1e-6
        
        # All should be non-negative
        assert bill['total_bill'] == power_level * 6.6148 + power_level * (39.95 + 6.97 + 2.01)

    def test_bill_cycle_reporting(self):
        """Test with full month billing period."""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 7, 1)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series for full month
        power_level = 10
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)  # 30 kW constant import
        
        bill_series = tariff_model.compute_bill_series(power_series)

        assert len(bill_series) == 1, "Should have exactly one billing cycle for full month"

        required_columns = ['demand_charge', 'energy_charge', 'total_bill']
        assert sorted(bill_series.columns) == sorted(required_columns), "Missing required columns in bill series"

        assert np.isclose(bill_series.iloc[0]['demand_charge'], power_level * (39.95 + 6.97 + 2.01))
        assert np.isclose(bill_series.iloc[0]['energy_charge'], power_level * 30 * 6.6148)
        assert np.isclose(bill_series.iloc[0]['total_bill'], power_level * 30 * 6.6148 + power_level * (39.95 + 6.97 + 2.01))
        

    def test_billing_period_spanning_months(self):
        """Test billing period spanning multiple months."""
        start_date = date(2024, 5, 15)  # Mid-January
        end_date = date(2024, 7, 15)    # Mid-March

        days_in_month = [31, 30, 31]  # May, June, July

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series spanning multiple months
        power_level = 10
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(power_level, index=time_index)  # 25 kW constant import
        
        bill_series = tariff_model.compute_bill_series(power_series)

        ref_series_values = np.array([[ 399.5   ,  666.0362, 1065.5362],
                                       [ 489.3   , 1984.44  , 2473.74  ],
                                       [ 489.3   ,  926.072 , 1415.372 ]])

        # Should have multiple billing cycles
        assert np.all(np.isclose(bill_series.values, ref_series_values, atol=1e-4)), "Bill series values do not match reference"
        bill_total = tariff_model.compute_total_bill(power_series)
        assert np.isclose(bill_total['total_bill'], bill_series['total_bill'].sum(), atol=1e-4), "Total bill does not match reference"
