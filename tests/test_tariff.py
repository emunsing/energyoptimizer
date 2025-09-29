import pandas as pd
import pytest
import matplotlib.pyplot as plt
from datetime import date, datetime
from src.energyoptimizer.tariff.tariff_utils import TariffModel

RATE_CODES = ['PGE_B_19_R']

class TestTariffModel:
    """Test suite for the TariffModel class."""
    
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
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)

        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)

        # Create a simple power series (constant import)
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(10.0, index=time_index)  # 10 kW constant import
        
        energy_charge = tariff_model.compute_energy_charge(power_series)
        
        # Should be positive (cost)
        assert energy_charge > 0, "Energy charge should be positive for import"
        
        # Should be reasonable magnitude (10 kW * 24 hours * rate)
        assert 0 < energy_charge < 1000, f"Energy charge {energy_charge} seems unreasonable"
    
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
        assert energy_charge <= 0, "Energy charge should be negative or zero for export"
    
    def test_compute_demand_charge_basic(self):
        """Test basic demand charge computation."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create a power series with peak demand
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(50.0, index=time_index)  # 50 kW constant import
        
        demand_charge = tariff_model.compute_demand_charge(power_series)
        
        # Should be positive
        assert demand_charge > 0, "Demand charge should be positive"
        
        # Should be reasonable magnitude
        assert 0 < demand_charge < 10000, f"Demand charge {demand_charge} seems unreasonable"
    
    def test_compute_total_bill(self):
        """Test total bill computation."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create a simple power series
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(10.0, index=time_index)
        
        bill = tariff_model.compute_total_bill(power_series)
        
        # Should have required fields
        assert 'demand_charge' in bill
        assert 'energy_charge' in bill
        assert 'total_bill' in bill
        
        # Total should equal sum of components
        assert abs(bill['total_bill'] - (bill['demand_charge'] + bill['energy_charge'])) < 1e-6
        
        # All should be non-negative
        assert bill['demand_charge'] >= 0
        assert bill['total_bill'] >= 0
    
    def test_power_series_less_than_month(self):
        """Test with power series less than a month."""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 6, 30)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series for just one week
        time_index = pd.date_range('2024-06-01', '2024-06-07', freq='15min', inclusive='left')
        power_series = pd.Series(20.0, index=time_index)  # 20 kW constant import
        
        bill = tariff_model.compute_total_bill(power_series)
        
        # Should still compute charges
        assert bill['total_bill'] > 0
        assert bill['energy_charge'] > 0
        # Demand charge might be zero if no peak periods in the week
    
    def test_full_month_billing_period(self):
        """Test with full month billing period."""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 6, 30)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series for full month
        time_index = pd.date_range('2024-06-01', '2024-06-30', freq='15min', inclusive='left')
        power_series = pd.Series(30.0, index=time_index)  # 30 kW constant import
        
        bill_series = tariff_model.compute_bill_series(power_series)
        
        # Should have at least one billing cycle
        assert len(bill_series) > 0
        
        # Each row should have required columns
        required_columns = ['demand_charge', 'energy_charge', 'total_bill', 'cycle_start', 'cycle_end']
        for col in required_columns:
            assert col in bill_series.columns
        
        # All charges should be non-negative
        assert (bill_series['demand_charge'] >= 0).all()
        assert (bill_series['total_bill'] >= 0).all()
    
    def test_billing_period_spanning_months(self):
        """Test billing period spanning multiple months."""
        start_date = date(2024, 1, 15)  # Mid-January
        end_date = date(2024, 3, 15)    # Mid-March
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series spanning multiple months
        time_index = pd.date_range('2024-01-15', '2024-03-15', freq='15min', inclusive='left')
        power_series = pd.Series(25.0, index=time_index)  # 25 kW constant import
        
        bill_series = tariff_model.compute_bill_series(power_series)
        
        # Should have multiple billing cycles
        assert len(bill_series) >= 2, "Should have at least 2 billing cycles for multi-month period"
        
        # Each billing cycle should be valid
        for _, row in bill_series.iterrows():
            assert row['cycle_start'] <= row['cycle_end']
            assert row['total_bill'] >= 0
    
    def test_demand_charge_peak_periods(self):
        """Test demand charge computation during peak periods."""
        start_date = date(2024, 6, 1)  # Summer month
        end_date = date(2024, 6, 30)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series with high demand during peak hours (4-9 PM)
        time_index = pd.date_range('2024-06-01', '2024-06-30', freq='15min', inclusive='left')
        power_series = pd.Series(0.0, index=time_index)  # Start with zero
        
        # Set high demand during peak hours
        peak_mask = time_index.hour.isin([16, 17, 18, 19, 20])
        power_series.loc[peak_mask] = 100.0  # 100 kW during peak
        
        demand_charge = tariff_model.compute_demand_charge(power_series)
        
        # Should have significant demand charge due to peak period
        assert demand_charge == 4692

    def test_export_revenue_calculation(self):
        """Test export revenue calculation when export rates are available."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 2)
        
        tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
        
        # Create power series with export (negative values)
        time_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
        power_series = pd.Series(-10.0, index=time_index)  # 10 kW constant export
        
        energy_charge = tariff_model.compute_energy_charge(power_series)
        
        # For PGE_B_19_R, export rates are null, so should be zero
        assert energy_charge == 0, "Energy charge should be zero when export rates are null"
    
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
        power_series.loc[day_mask] = 20.0   # 20 kW import during day
        power_series.loc[~day_mask] = -5.0  # 5 kW export during night
        
        bill = tariff_model.compute_total_bill(power_series)
        
        # Should have positive total bill (net import)
        assert bill['total_bill'] > 0
        assert bill['energy_charge'] > 0  # Net import should result in cost

