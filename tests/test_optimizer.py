import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from src.energyoptimizer.optimizers import (
    tou_optimization, 
    self_consumption, 
    tou_endogenous_sizing_optimization,
    OptimizationInputs,
    OptimizerOutputs
)
from src.energyoptimizer.optimization_runner import OptimizationType
from src.energyoptimizer.tariff.tariff_utils import TariffModel


@pytest.fixture
def sample_tariff_model():
    """Create a TariffModel for testing."""
    from datetime import date
    return TariffModel('tariffs.yaml', 'PGE_B_19_R', '2025-01-01', '2035-01-01')


def sample_site_data(start_date, end_date, freq):
    """Create sample site data for testing."""
    # Create a simple time series for one month
    np.random.seed(0)
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create realistic solar and load profiles
    n_periods = len(index)
    
    # Solar profile: peak at noon, zero at night, with seasonal variation
    hour_of_day = index.hour + index.minute / 60
    day_of_year = index.dayofyear
    
    # Seasonal variation: peak in June (~day 172), minimum in December (~day 355)
    # Amplitude varies from ~4 kW in December to ~7 kW in June
    seasonal_amplitude = 4.0 + 3.0 * np.sin((day_of_year - 80) * 2 * np.pi / 365)  # 80 days offset for June peak
    
    # Daily solar profile with seasonal amplitude
    solar_profile = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * seasonal_amplitude)
    
    # Load profile: higher during day, lower at night
    load_profile = 3.0 + 2.0 * np.sin((hour_of_day - 6) * np.pi / 12) + np.random.normal(0, 0.2, n_periods)
    load_profile = np.maximum(0.5, load_profile)  # Minimum 0.5 kW load
    
    # Non-circuit load (constant)
    non_circuit_load = np.full(n_periods, 1.0)
    
    site_data = pd.DataFrame({
        'solar': solar_profile,
        'load': load_profile,
        'non_circuit_load': non_circuit_load
    }, index=index)
    
    return site_data

def site_data_and_tariff_model(start_date, end_date, freq):
    site_data = sample_site_data(start_date, end_date, freq)
    tariff_model = TariffModel('tariffs.yaml', 'PGE_B_19_R', start_date, end_date)
    return OptimizationInputs(
        site_data=site_data,
        tariff_model=tariff_model,
        batt_rt_eff=0.85,
        batt_block_e_max=13.5,
        batt_block_p_max=5.0,
        backup_reserve=0.2,
        circuit_import_kw_limit=100.0,
        circuit_export_kw_limit=-100.0,
        site_import_kw_limit=100.0,
        site_export_kw_limit=-100.0,
        solar_annualized_cost_per_kw=0.15,
        batt_annualized_cost_per_unit=1000.0,
        integer_problem=False
    )


@pytest.fixture
def one_month_optimization_inputs():
    return site_data_and_tariff_model(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2025-01-31 23:45:00-08:00'), '15min')


@pytest.fixture
def one_year_optimization_inputs():
    return site_data_and_tariff_model(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2025-12-31 23:45:00-08:00'), '1h')   


@pytest.fixture
def ten_years_optimization_inputs():
    return sample_site_data(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2034-12-31 23:45:00-08:00'), '1h')


@pytest.fixture
def optimizer_functions():
    """Return all optimizer functions for parameterized testing."""
    return [
        (tou_optimization, "tou_optimization"),
        (self_consumption, "self_consumption"),
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
    ]


class TestOptimizerIntegration:
    """Integration tests for all optimizers."""
    
    def test_optimization_inputs_creation(self, one_month_optimization_inputs):
        sample_optimization_inputs = one_month_optimization_inputs
        """Test that OptimizationInputs can be created with sample data."""
        assert sample_optimization_inputs.site_data is not None
        assert sample_optimization_inputs.tariff_model is not None
        assert len(sample_optimization_inputs.site_data) > 0
        assert 'solar' in sample_optimization_inputs.site_data.columns
        assert 'load' in sample_optimization_inputs.site_data.columns
    
    @pytest.mark.parametrize("optimizer_func,optimizer_name", [
        (tou_optimization, "tou_optimization"),
        (self_consumption, "self_consumption"),
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
    ])
    def test_optimizer_runs_successfully(self, optimizer_func, optimizer_name, sample_optimization_inputs):
        """Test that each optimizer runs without errors."""
        result = optimizer_func(sample_optimization_inputs)
        
        # Check that result is OptimizerOutputs
        assert isinstance(result, OptimizerOutputs)
        assert hasattr(result, 'results_df')
        assert hasattr(result, 'sizing_results')
        
        # Check that results_df is a DataFrame
        assert isinstance(result.results_df, pd.DataFrame)
        assert len(result.results_df) > 0
        
        # Check that results_df has expected columns
        if optimizer_name == "tou_endogenous_sizing_optimization":
            # Endogenous sizing optimizer has different columns
            expected_columns = ['P_batt', 'P_grid', 'E']
        else:
            expected_columns = ['P_batt', 'P_grid', 'E', 'solar_post_curtailment']
        
        for col in expected_columns:
            assert col in result.results_df.columns, f"Missing column: {col}"
    
    @pytest.mark.parametrize("optimizer_func,optimizer_name", [
        (tou_optimization, "tou_optimization"),
        (self_consumption, "self_consumption"),
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
    ])
    def test_optimizer_results_consistency(self, optimizer_func, optimizer_name, sample_optimization_inputs):
        """Test that optimizer results are consistent and reasonable."""
        result = optimizer_func(sample_optimization_inputs)
        df = result.results_df
        
        # Check that all values are finite
        for col in df.columns:
            assert df[col].notna().all(), f"NaN values found in {col}"
            assert np.isfinite(df[col]).all(), f"Non-finite values found in {col}"
        
        # Check battery energy bounds
        if 'E' in df.columns:
            assert (df['E'] >= 0).all(), "Battery energy should be non-negative"
            assert (df['E'] <= 13.5).all(), "Battery energy should not exceed capacity"
        
        # Check battery power bounds
        if 'P_batt' in df.columns:
            assert (df['P_batt'] >= -5.0).all(), "Battery charge power should not exceed limit"
            assert (df['P_batt'] <= 5.0).all(), "Battery discharge power should not exceed limit"
    
    def test_endogenous_sizing_returns_sizing_results(self, sample_optimization_inputs):
        """Test that endogenous sizing optimizer returns sizing results."""
        result = tou_endogenous_sizing_optimization(sample_optimization_inputs)
        
        # Check that sizing results are present
        sizing_results = result.get_sizing_results()
        assert 'n_batts' in sizing_results
        assert 's_size_kw' in sizing_results
        
        # Check that sizing results are reasonable
        assert sizing_results['n_batts'] >= 0
        assert sizing_results['s_size_kw'] >= 0
    

    def test_regular_optimizers_no_sizing_results(self, sample_optimization_inputs):
        """Test that regular optimizers don't return sizing results."""
        for optimizer_func in [tou_optimization, self_consumption]:
            result = optimizer_func(sample_optimization_inputs)
            sizing_results = result.get_sizing_results()
            assert len(sizing_results) == 0, f"{optimizer_func.__name__} should not return sizing results"

    
    def test_optimizer_outputs_methods(self, sample_optimization_inputs):
        """Test that OptimizerOutputs methods work correctly."""
        result = tou_optimization(sample_optimization_inputs)
        
        # Test get_results method
        results_df = result.get_results()
        assert isinstance(results_df, pd.DataFrame)
        assert results_df.equals(result.results_df)
        
        # Test get_sizing_results method
        sizing_results = result.get_sizing_results()
        assert isinstance(sizing_results, dict)
        assert sizing_results == result.sizing_results

class TestOptimizerClockFrequencies:
    """Test optimizers with different clock frequencies."""
    
    def test_one_month_optimization(self, sample_optimization_inputs):
        """Test optimization with one-month data (no clock needed)."""
        for optimizer_func in [tou_optimization, self_consumption]:
            result = optimizer_func(sample_optimization_inputs)
            assert isinstance(result, OptimizerOutputs)
            assert len(result.results_df) == len(sample_optimization_inputs.site_data)
    
    def test_one_year_with_monthly_clock(self, one_year_optimization_inputs):
        """Test one-year optimization with monthly clock frequency."""
        from src.energyoptimizer.optimization_runner import OptimizationClock
        
        # Create monthly optimization clock
        clock = OptimizationClock(
            frequency='1M',  # Monthly frequency
            horizon=pd.DateOffset(months=1),  # 1 month horizon
            lookback=pd.DateOffset(months=1)  # 1 month lookback
        )
        
        # Test that clock generates appropriate intervals
        start = one_year_optimization_inputs.site_data.index[0]
        end = one_year_optimization_inputs.site_data.index[-1]
        intervals = clock.get_intervals(start, end)
        
        # Should have 12 monthly intervals for a year
        assert len(intervals) == 12, f"Expected 12 monthly intervals, got {len(intervals)}"
        
        # Test that intervals are properly spaced
        for i, (optimize_at, data_from, data_until) in enumerate(intervals):
            assert data_from <= optimize_at <= data_until, f"Invalid interval {i}"
            if i > 0:
                # Check that intervals are approximately monthly
                prev_optimize = intervals[i-1][0]
                time_diff = optimize_at - prev_optimize
                assert 25 <= time_diff.days <= 35, f"Intervals should be ~monthly, got {time_diff.days} days"
    
    def test_ten_years_with_yearly_clock(self, ten_years_optimization_inputs):
        """Test ten-year optimization with yearly clock frequency."""
        from src.energyoptimizer.optimization_runner import OptimizationClock
        
        # Create yearly optimization clock
        clock = OptimizationClock(
            frequency='1Y',  # Yearly frequency
            horizon=pd.DateOffset(years=1),  # 1 year horizon
            lookback=pd.DateOffset(years=1)  # 1 year lookback
        )
        
        # Test that clock generates appropriate intervals
        start = ten_years_optimization_inputs.site_data.index[0]
        end = ten_years_optimization_inputs.site_data.index[-1]
        intervals = clock.get_intervals(start, end)
        
        # Should have 10 yearly intervals for 10 years
        assert len(intervals) == 10, f"Expected 10 yearly intervals, got {len(intervals)}"
        
        # Test that intervals are properly spaced
        for i, (optimize_at, data_from, data_until) in enumerate(intervals):
            assert data_from <= optimize_at <= data_until, f"Invalid interval {i}"
            if i > 0:
                # Check that intervals are approximately yearly
                prev_optimize = intervals[i-1][0]
                time_diff = optimize_at - prev_optimize
                assert 360 <= time_diff.days <= 370, f"Intervals should be ~yearly, got {time_diff.days} days"
    
    def test_optimizer_with_clock_intervals(self, one_year_optimization_inputs):
        """Test that optimizers work with clock-generated intervals."""
        from src.energyoptimizer.optimization_runner import OptimizationClock
        
        clock = OptimizationClock(
            frequency='1M',
            horizon=pd.DateOffset(months=1),
            lookback=pd.DateOffset(months=1)
        )
        
        start = one_year_optimization_inputs.site_data.index[0]
        end = one_year_optimization_inputs.site_data.index[-1]
        intervals = clock.get_intervals(start, end)
        
        # Test optimization on first interval
        optimize_at, data_from, data_until = intervals[0]
        interval_data = one_year_optimization_inputs.site_data.loc[data_from:data_until]
        
        # Create new optimization inputs for this interval
        interval_inputs = OptimizationInputs(
            site_data=interval_data,
            tariff_model=one_year_optimization_inputs.tariff_model,
            batt_rt_eff=one_year_optimization_inputs.batt_rt_eff,
            batt_block_e_max=one_year_optimization_inputs.batt_block_e_max,
            batt_block_p_max=one_year_optimization_inputs.batt_block_p_max,
            backup_reserve=one_year_optimization_inputs.backup_reserve,
            circuit_import_kw_limit=one_year_optimization_inputs.circuit_import_kw_limit,
            circuit_export_kw_limit=one_year_optimization_inputs.circuit_export_kw_limit,
            site_import_kw_limit=one_year_optimization_inputs.site_import_kw_limit,
            site_export_kw_limit=one_year_optimization_inputs.site_export_kw_limit,
            solar_annualized_cost_per_kw=one_year_optimization_inputs.solar_annualized_cost_per_kw,
            batt_annualized_cost_per_unit=one_year_optimization_inputs.batt_annualized_cost_per_unit,
            integer_problem=one_year_optimization_inputs.integer_problem
        )
        
        # Test that optimizers work with interval data
        for optimizer_func in [tou_optimization, self_consumption]:
            result = optimizer_func(interval_inputs)
            assert isinstance(result, OptimizerOutputs)
            assert len(result.results_df) == len(interval_data)


if __name__ == "__main__":
    pytest.main([__file__])
