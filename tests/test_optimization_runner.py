import attrs
import numpy as np
import pandas as pd
import pytest

from src.energyoptimizer.batteryopt_interface import OptimizationClock, TariffSpec, DesignSpec, FinancialSpec, \
    GeneralAssumptions, ScenarioSpec, OptimizationRunnerInputs, OptimizationType
from src.energyoptimizer.optimization_runner import OptimizationRunner
from tests.test_optimizer import TZ
from tests.test_utils import sample_site_data


@pytest.fixture
def optimizer_clock_freq_list():
    return [
        OptimizationClock(frequency='2W-SUN', horizon=pd.DateOffset(months=1), lookback=None),
        OptimizationClock(frequency='M', horizon=pd.DateOffset(months=2), lookback=None),
        OptimizationClock(frequency='Q', horizon=pd.DateOffset(months=6), lookback=None),
    ]


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


sample_optimizer_clock_list = [OptimizationClock(frequency='W-SUN', horizon=None, lookback=None),
                               OptimizationClock(frequency='W-SUN', horizon=pd.DateOffset(days=14), lookback=None),
                              OptimizationClock(frequency='W-SUN', horizon=None, lookback=pd.DateOffset(days=7)),
                              OptimizationClock(frequency='W-SUN', horizon=pd.DateOffset(days=14), lookback=pd.DateOffset(days=7)),
                              ]
demand_charge_optimizer_clock_list = [
        OptimizationClock(frequency='MS', horizon=pd.DateOffset(months=2), lookback=None),  # Use starting freqs
        OptimizationClock(frequency='QS', horizon=pd.DateOffset(months=4), lookback=None),
        OptimizationClock(frequency='1YS', horizon=None, lookback=None),
    ]


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

    @pytest.mark.parametrize('clock_invariant_optimizer,optimizer_clock_list',
                             [
                                 ("tou_optimization", sample_optimizer_clock_list),
                                 ("SUBPANEL_SELF_CONSUMPTION", sample_optimizer_clock_list),
                                 ("demand_charge_tou_optimization", demand_charge_optimizer_clock_list),
                             ]
                             )
    def test_optimization_runner_clock_types(self, clock_invariant_optimizer, optimizer_clock_list):
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

        for clock in optimizer_clock_list:
            inputs = attrs.evolve(inputs, optimization_clock=clock)
            runner = OptimizationRunner(inputs)
            results = runner.run_optimization()
            result_df = results.results_df
            assert result_df.index[0] == general_assumptions.start_date
            assert result_df.index[-1] >= general_assumptions.end_date - pd.tseries.frequencies.to_offset(general_assumptions.study_resolution)
            reference_results[clock_invariant_optimizer] = results
            if previous_results is None:
                previous_results = results
                continue
            else:
                # Check that results are identical regardless of clock settings
                assert np.allclose(previous_results.results_df.abs().sum(), results.results_df.abs().sum(), rtol=5e-2)  # Allow 5% tolerance for end conditions when using a 1-week overlap
                previous_results = results

    @pytest.mark.parametrize('clock_invariant_optimizer,optimizer_clock_list',
                             [
                                 ("tou_optimization", sample_optimizer_clock_list),
                                  ("SUBPANEL_SELF_CONSUMPTION", sample_optimizer_clock_list),
                                  ("demand_charge_tou_optimization", demand_charge_optimizer_clock_list),
                             ]
                             )
    def test_optimization_runner_clock_intervals(self, clock_invariant_optimizer, optimizer_clock_list):
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

        for clock in optimizer_clock_list:
            inputs = attrs.evolve(inputs, optimization_clock=clock)
            runner = OptimizationRunner(inputs)
            results = runner.run_optimization()
            result_df = results.results_df
            assert result_df.index[0] == self.default_general_assumptions.start_date
            assert result_df.index[-1] >= self.default_general_assumptions.end_date - pd.tseries.frequencies.to_offset(self.default_general_assumptions.study_resolution)
            reference_results[clock_invariant_optimizer] = results
            if previous_results is None:
                previous_results = results
                continue
            else:
                # Check that results are identical regardless of clock settings
                assert np.allclose(previous_results.results_df.abs().sum(), results.results_df.abs().sum(), rtol=1e-1)  # Allow 5% tolerance for end conditions when using a 1-week overlap
                previous_results = results
