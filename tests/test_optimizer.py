import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
from unittest.mock import Mock
from .test_utils import sample_site_data

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


# Fixture-of-fixtures to iterate over your three input sets
@pytest.fixture(
    params=[
        "one_month_optimization_inputs",
        "one_year_optimization_inputs",
        "ten_years_optimization_inputs",
    ]
)
def sample_optimization_inputs(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def optimizer_functions():
    """Return all optimizer functions for parameterized testing."""
    return [
        (tou_optimization, "tou_optimization"),
        (self_consumption, "self_consumption"),
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
    ]


def test_optimization_inputs_creation(one_year_optimization_inputs):
    sample_optimization_inputs = one_year_optimization_inputs
    """Test that OptimizationInputs can be created with sample data."""
    assert sample_optimization_inputs.site_data.shape[0] == 8760

    ref_cols = sorted(['solar', 'load', 'non_circuit_load'])
    assert ref_cols == sorted(list(sample_optimization_inputs.site_data.columns))

    solar = sample_optimization_inputs.site_data['solar']
    solar_day1 = solar.iloc[0:24]
    assert solar_day1.max() <= 7.0
    assert solar_day1.max() >= 4.0
    assert solar_day1.min() == 0.0
    monthly_peak_solar = solar.resample('M').max()
    assert monthly_peak_solar.max() / monthly_peak_solar.min() > 1.5  # Require Seasonal variation

    assert sample_optimization_inputs.tariff_model is not None

@pytest.mark.slow
@pytest.mark.parametrize("optimizer_func,optimizer_name", [
    (tou_optimization, "tou_optimization"),
    (self_consumption, "self_consumption"),
    (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
])
def test_bare_optimizer_func(optimizer_func, optimizer_name, sample_optimization_inputs):
    """Test that each optimizer runs without errors."""
    start_time = time.time()
    result = optimizer_func(sample_optimization_inputs)
    print("Optimizer:", optimizer_name, "Time taken (s):", time.time() - start_time)

    # Check that result is OptimizerOutputs
    assert isinstance(result, OptimizerOutputs)

    # Check that results_df is a DataFrame
    assert isinstance(result.results_df, pd.DataFrame)
    expected_columns = ["P_batt", "P_grid", "E", "solar_used_directly"]
    assert np.all([col in result.results_df.columns for col in expected_columns])
    assert result.results_df.shape[0] == sample_optimization_inputs.site_data.shape[0]

    """
    Accuracy checks:
    - Curtailed solar <= input solar
    - Battery charge/discharge within limits
    - Grid import/export within limits
    - Grid + Battery + Solar = Load + Non-circuit load
    - Battery SoC within limits
    """


def test_optimizer_clock_no_horizon_or_lookback():
    pass

def test_optimizer_clock_lookback_no_horizon():
    pass

def test_optimizer_clock_horizon_no_lookback():
    pass

def test_optimizer_clock_horizon_and_lookback():
    pass


@pytest.fixture
def sample_optimization_runner():
    pass



class TestO:

    def test_clock_windows(self, ):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
