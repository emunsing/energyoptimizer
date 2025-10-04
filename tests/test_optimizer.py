import pytest
import pandas as pd
import numpy as np
import time
from .test_utils import sample_site_data, simple_optimization_inputs
import attrs
from src.energyoptimizer.batteryopt_interface import (
    DesignSpec, FinancialSpec, GeneralAssumptions, TariffSpec, ScenarioSpec,
    DesignInputs, FinancialModelInputs
)
from src.energyoptimizer.optimizers import (
    tou_optimization, 
    single_panel_self_consumption,
    subpanel_self_consumption,
    tou_endogenous_sizing_optimization,
    OptimizerOutputs
)
from src.energyoptimizer.optimization_runner import OptimizationRunner
from src.energyoptimizer.batteryopt_interface import OptimizationClock
from src.energyoptimizer.tariff.tariff_utils import TariffModel
from src.energyoptimizer.optimization_runner import OptimizationRunnerInputs, OptimizationType

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
        (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
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
    (tou_endogenous_sizing_optimization, "tou_endogenous_sizing_optimization")
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
    assert np.all(results_df['E'] <= sample_optimization_inputs.batt_block_e_max * sizing_results['n_batt_blocks'])


# Clock types: Different frequencies:
@pytest.fixture
def optimizer_clock_freq_list():
    return [
        OptimizationClock(frequency='2W-SUN', horizon=pd.DateOffset(months=1), lookback=None),
        OptimizationClock(frequency='M', horizon=pd.DateOffset(months=2), lookback=None),
        OptimizationClock(frequency='Q', horizon=pd.DateOffset(months=6), lookback=None),
    ]


# Clock types: no/horizons
@pytest.fixture
def optimizer_clock_no_horizon_or_lookback():
    return OptimizationClock(frequency='W-SUN', horizon=None, lookback=None)

@pytest.fixture
def optimizer_clock_lookback_no_horizon():
    return OptimizationClock(frequency='W-SUN', horizon=None, lookback=pd.DateOffset(days=7))

@pytest.fixture
def optimizer_clock_horizon_no_lookback():
    return OptimizationClock(frequency='W-SUN', horizon=pd.DateOffset(days=14), lookback=None)

@pytest.fixture
def optimizer_clock_horizon_and_lookback():
    return OptimizationClock(frequency='W-SUN', horizon=pd.DateOffset(days=14), lookback=pd.DateOffset(days=7))

@pytest.fixture(
    params=[
        "optimizer_clock_no_horizon_or_lookback",
        "optimizer_clock_lookback_no_horizon",
        "optimizer_clock_horizon_no_lookback",
        "optimizer_clock_horizon_and_lookback",
    ]
)
def sample_optimization_clock(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def sample_optimization_clock_list(request):
    return [
        request.getfixturevalue("optimizer_clock_no_horizon_or_lookback"),
        request.getfixturevalue("optimizer_clock_lookback_no_horizon"),
        request.getfixturevalue("optimizer_clock_horizon_no_lookback"),
        request.getfixturevalue("optimizer_clock_horizon_and_lookback"),
    ]


class TestOptimizationRunner:

    one_year_site_data = sample_site_data('2023-01-01', '2024-01-01', freq='1H', tz=TZ)

    default_tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
    default_design_spec = DesignSpec(solar_data_source="upload",
                                     solar_data=one_year_site_data[['solar']],
                                     circuit_load_data_source="upload",
                                     circuit_load_data=one_year_site_data[['der_subpanel_load']],
                                     non_circuit_load_data_source="upload",
                                     non_circuit_load_data=one_year_site_data[['main_panel_load']],
                                     )
    default_financial_spec = FinancialSpec()
    default_general_assumptions = GeneralAssumptions(start_date='2026-01-01',
                                                     study_years=1,
                                                     )
    default_scenario_spec = ScenarioSpec(
        general_assumptions=default_general_assumptions,
        design_spec=default_design_spec,
        tariff_spec=default_tariff_spec,
        financial_spec=default_financial_spec
    )

    @pytest.mark.parametrize('clock_invariant_optimizer',
                             [
                                 "tou_optimization",
                                 "SUBPANEL_SELF_CONSUMPTION"
                             ]
                             )
    def test_optimization_runner_clock_types(self, clock_invariant_optimizer, sample_optimization_clock_list):
        # Run this with a shorter date range for speed; we're using a 1-week clock
        general_assumptions = GeneralAssumptions(start_date='2026-01-01', end_date='2026-03-01')
        scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=self.default_design_spec,
            tariff_spec=self.default_tariff_spec,
            financial_spec=self.default_financial_spec
        )
        design_inputs = scenario_spec.build_design_inputs()
        financial_inputs = scenario_spec.build_financial_model_inputs()

        inputs = OptimizationRunnerInputs(
            optimization_type=OptimizationType[clock_invariant_optimizer.upper()],
            optimization_start=general_assumptions.start_date,
            optimization_end=general_assumptions.end_date,
            design_inputs=design_inputs,
            financial_model_inputs=financial_inputs,
        )

        reference_results = {}
        previous_results = None

        for clock in sample_optimization_clock_list:
            inputs = attrs.evolve(inputs, optimization_clock=clock)
            runner = OptimizationRunner(inputs)
            results = runner.run_optimization()
            reference_results[clock_invariant_optimizer] = results
            if previous_results is None:
                previous_results = results
                continue
            else:
                # Check that results are identical regardless of clock settings
                assert np.allclose(previous_results.results_df.abs().sum(), results.results_df.abs().sum(), rtol=1e-2)  # Allow 5% tolerance for end conditions when using a 1-week overlap
                previous_results = results

    @pytest.mark.parametrize('clock_invariant_optimizer',
                             [
                                 "tou_optimization",
                                 "SUBPANEL_SELF_CONSUMPTION"
                             ]
                             )
    def test_optimization_runner_clock_intervals(self, clock_invariant_optimizer, optimizer_clock_freq_list):
        # Run this with a shorter date range for speed; we're using a 1-week clock
        design_inputs = self.default_scenario_spec.build_design_inputs()
        financial_inputs = self.default_scenario_spec.build_financial_model_inputs()

        inputs = OptimizationRunnerInputs(
            optimization_type=OptimizationType[clock_invariant_optimizer.upper()],
            optimization_start=self.default_general_assumptions.start_date,
            optimization_end=self.default_general_assumptions.end_date,
            design_inputs=design_inputs,
            financial_model_inputs=financial_inputs,
        )

        reference_results = {}
        previous_results = None

        for clock in optimizer_clock_freq_list:
            inputs = attrs.evolve(inputs, optimization_clock=clock)
            runner = OptimizationRunner(inputs)
            results = runner.run_optimization()
            reference_results[clock_invariant_optimizer] = results
            if previous_results is None:
                previous_results = results
                continue
            else:
                # Check that results are identical regardless of clock settings
                assert np.allclose(previous_results.results_df.abs().sum(), results.results_df.abs().sum(), rtol=1e-1)  # Allow 5% tolerance for end conditions when using a 1-week overlap
                previous_results = results


if __name__ == "__main__":
    pytest.main([__file__])
