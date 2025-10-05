import pytest
import pandas as pd
import numpy as np
import time
from .test_utils import simple_optimization_inputs
from src.energyoptimizer.optimizers import (
    tou_optimization,
    demand_charge_tou_optimization,
    demand_charge_tou_endogenous_sizing_optimization,
    subpanel_self_consumption,
    tou_endogenous_sizing_optimization,
    OptimizerOutputs
)
from src.energyoptimizer.tariff.tariff_utils import TariffModel

TZ = 'US/Pacific'

@pytest.fixture
def sample_tariff_model():
    """Create a TariffModel for testing."""
    return TariffModel('tariffs.yaml', 'PGE_B_19_R', '2025-01-01', '2035-01-01')


@pytest.fixture
def one_month_optimization_inputs():
    return simple_optimization_inputs(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2025-01-31 23:45:00-08:00'), '15min')


@pytest.fixture
def one_year_optimization_inputs():
    return simple_optimization_inputs(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2025-12-31 23:45:00-08:00'), '1h')


@pytest.fixture
def ten_years_optimization_inputs():
    return simple_optimization_inputs(pd.Timestamp('2025-01-01 00:00:00-08:00'), pd.Timestamp('2034-12-31 23:45:00-08:00'), '1h')


# Fixture-of-fixtures to iterate over your three input sets
@pytest.fixture(
    params=[
        "one_month_optimization_inputs",
        "one_year_optimization_inputs",
        # "ten_years_optimization_inputs",  # Slow - but passes as of 10/3/2025
    ]
)
def sample_optimization_inputs(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def sample_optimization_input_list(request):
    return [
        request.getfixturevalue("one_month_optimization_inputs"),
        request.getfixturevalue("one_year_optimization_inputs"),
        request.getfixturevalue("ten_years_optimization_inputs"),
    ]


@pytest.fixture
def optimizer_function_list():
    """Return all optimizer functions for parameterized testing."""
    return [
        (tou_optimization, "tou_optimization"),
        (subpanel_self_consumption, "self_consumption"),
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization"),
        (demand_charge_tou_optimization, "demand_charge_tou_optimization"),
        (demand_charge_tou_endogenous_sizing_optimization, "demand_charge_tou_endogenous_sizing"),
    ]


def test_optimization_inputs_creation(one_year_optimization_inputs):
    sample_optimization_inputs = one_year_optimization_inputs
    """Test that OptimizationInputs can be created with sample data."""
    assert sample_optimization_inputs.site_data.shape[0] == 8760

    ref_cols = sorted(['main_panel_load', 'solar', 'der_subpanel_load'])
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
    (subpanel_self_consumption, "subpanel_self_consumption"),
    (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization"),
    (demand_charge_tou_optimization, "demand_charge_tou_optimization"),
    (demand_charge_tou_endogenous_sizing_optimization, "demand_charge_tou_endogenous_sizing"),
])
def test_bare_optimizer_func(optimizer_func, optimizer_name, sample_optimization_inputs):
    start_time = time.time()
    result = optimizer_func(sample_optimization_inputs)
    print("Optimizer:", optimizer_name, "Time taken (s):", time.time() - start_time)

    # Check that result is OptimizerOutputs
    assert isinstance(result, OptimizerOutputs)

    # Check that results_df is a DataFrame
    results_df = result.results_df
    sizing_results = result.sizing_results
    assert isinstance(results_df, pd.DataFrame)
    expected_columns = ["P_batt", "P_grid", "E", "solar_post_curtailment"]
    assert np.all([col in results_df.columns for col in expected_columns])
    assert results_df.shape[0] == sample_optimization_inputs.site_data.shape[0]

    assert np.all(results_df['solar_post_curtailment'] >= 0)
    assert (results_df['solar_post_curtailment'] - sample_optimization_inputs.site_data['solar'] * sizing_results['n_solar']).max() <= 1e-6  # Allow for numeric inaccuracy

    # Battery charge/discharge within limits
    # P_batt should be within [-batt_block_p_max, batt_block_p_max]
    assert np.all(results_df['P_batt'] >= -sample_optimization_inputs.batt_block_p_max)
    assert np.all(results_df['P_batt'] <= sample_optimization_inputs.batt_block_p_max)
    
    # Grid import/export within limits
    assert np.all(results_df['P_grid'] >= sample_optimization_inputs.site_export_kw_limit)
    assert np.all(results_df['P_grid'] <= sample_optimization_inputs.site_import_kw_limit)

    # P_grid should be within [circuit_export_kw_limit, circuit_import_kw_limit]
    assert np.all(results_df['P_subpanel'] >= sample_optimization_inputs.der_subpanel_export_kw_limit)
    assert np.all(results_df['P_subpanel'] <= sample_optimization_inputs.der_subpanel_import_kw_limit)
    
    # Energy balance: Grid + Battery + Solar = Load + Non-circuit load
    # Get load data from site_data
    site_data = sample_optimization_inputs.site_data
    total_load = site_data['der_subpanel_load'] + site_data['main_panel_load']
    energy_balance = results_df['P_grid'] + results_df['P_batt'] + results_df['solar_post_curtailment']
    # Allow small numerical tolerance for floating point precision
    assert np.allclose(energy_balance, total_load, rtol=1e-6, atol=1e-6)
    
    # Battery SoC within limits
    # E (battery energy/SoC) should be within [0, batt_block_e_max]
    assert np.all(results_df['E'] >= 0)
    assert np.all(results_df['E'] <= sample_optimization_inputs.batt_block_e_max * sizing_results['n_batt_blocks'] + 1e-6)


def test_demand_charge_optimization(sample_optimization_inputs):
    optimization_inputs = sample_optimization_inputs
    demand_charge_optimization_result = demand_charge_tou_optimization(optimization_inputs)
    assert demand_charge_optimization_result.status == 'optimal'
    p_grid = demand_charge_optimization_result.results_df['P_grid']
    tariff = optimization_inputs.tariff_model
    demand_charge_bill_cycles = tariff.compute_bill_series(p_grid)
    demand_charge_total_bill = tariff.compute_total_bill(p_grid)

    tou_optimization_result = tou_optimization(optimization_inputs)
    assert tou_optimization_result.status == 'optimal'
    p_grid_tou = tou_optimization_result.results_df['P_grid']
    tou_bill_cycles = tariff.compute_bill_series(p_grid_tou)
    tou_total_bill = tariff.compute_total_bill(p_grid_tou)

    assert demand_charge_total_bill['total_bill'] <= tou_total_bill['total_bill']
    assert np.all(demand_charge_bill_cycles['demand_charge'] <= tou_bill_cycles['demand_charge'])

def test_demand_charge_endogenous_sizing_optimization(sample_optimization_inputs):
    optimization_inputs = sample_optimization_inputs
    demand_charge_optimization_result = demand_charge_tou_endogenous_sizing_optimization(optimization_inputs)
    assert demand_charge_optimization_result.status == 'optimal'
    sizing_results = demand_charge_optimization_result.sizing_results
    assert sizing_results['n_batt_blocks'] >= 1
    assert sizing_results['n_solar'] >= 1

    p_grid = demand_charge_optimization_result.results_df['P_grid']
    tariff = optimization_inputs.tariff_model
    demand_charge_bill_cycles = tariff.compute_bill_series(p_grid)
    demand_charge_total_bill = tariff.compute_total_bill(p_grid)

    tou_optimization_result = tou_endogenous_sizing_optimization(optimization_inputs)
    assert tou_optimization_result.status == 'optimal'
    p_grid_tou = tou_optimization_result.results_df['P_grid']
    tou_bill_cycles = tariff.compute_bill_series(p_grid_tou)
    tou_total_bill = tariff.compute_total_bill(p_grid_tou)

    assert demand_charge_total_bill['total_bill'] <= tou_total_bill['total_bill']
    assert np.all(demand_charge_bill_cycles['demand_charge'] <= tou_bill_cycles['demand_charge'])


if __name__ == "__main__":
    pytest.main([__file__])
