import pytest
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from unittest.mock import Mock, MagicMock

from energyoptimizer.scenario_runner import (
    closest_n_elements, ResultSummary, BasicResultSummarizer, 
    ScenarioRunner, SizingSweepScenarioRunner, TopNScenarioRunner
)
from energyoptimizer.batteryopt_interface import (
    ScenarioSpec, GeneralAssumptions, DesignSpec, TariffSpec, FinancialSpec,
    DesignInputs, FinancialModelInputs, OptimizationType, OptimizationClock,
    OptimizationRunnerInputs, ProductCashFlows, PRODUCT_TO_SIZING_OUTPUT_MAP
)
from energyoptimizer.batteryopt_utils import MIN_DT
from energyoptimizer.optimizers import OptimizerOutputs
from energyoptimizer.tariff.tariff_utils import TariffModel
from tests.test_utils import sample_site_data
from tests.test_optimizer import TZ


# Shared fixtures that leverage existing helper functions and default specifications
class SharedTestFixtures:
    """Shared test fixtures that reuse existing helper functions and default specifications."""
    
    @staticmethod
    def create_default_scenario_spec(start_date='2023-01-01', study_years=1, end_date=None,
                                   min_battery_units=1, max_battery_units=3,
                                   min_solar_units=1, max_solar_units=5):
        """Create a scenario spec using the same pattern as TestOptimizationRunner."""
        site_data = sample_site_data('2023-01-01', '2024-01-01', freq='1H', tz=TZ)
        start_date = pd.Timestamp(start_date, tz=TZ)

        if end_date is not None:
            end_date = pd.Timestamp(end_date, tz=TZ)
            study_years = int(np.ceil((end_date - start_date).days / 365.25))
        
        general_assumptions = GeneralAssumptions(
            start_date=start_date,
            end_date=end_date,
            study_years=study_years,
            optimization_type='tou_optimization'
        )
        
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=site_data[['solar']],
            circuit_load_data_source="upload",
            circuit_load_data=site_data[['der_subpanel_load']],
            non_circuit_load_data_source="upload",
            non_circuit_load_data=site_data[['main_panel_load']],
            min_battery_units=min_battery_units,
            max_battery_units=max_battery_units,
            min_solar_units=min_solar_units,
            max_solar_units=max_solar_units
        )
        
        tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
        financial_spec = FinancialSpec(study_years=study_years)
        
        return ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
    
    @staticmethod
    def create_design_inputs_from_scenario_spec(scenario_spec):
        """Create design inputs from scenario spec using the build method."""
        return scenario_spec.build_design_inputs()
    
    @staticmethod
    def create_financial_inputs_from_scenario_spec(scenario_spec):
        """Create financial inputs from scenario spec using the build method."""
        return scenario_spec.build_financial_model_inputs()


# Shared fixtures
@pytest.fixture
def short_scenario_spec():
    """Short scenario spec for fast testing."""
    return SharedTestFixtures.create_default_scenario_spec(
        start_date='2026-01-01',
        end_date='2026-03-01',
        min_battery_units=1, 
        max_battery_units=5,
        min_solar_units=1, 
        max_solar_units=5
    )

@pytest.fixture
def oneyear_scenario_spec():
    return SharedTestFixtures.create_default_scenario_spec(
        start_date='2026-01-01',
        end_date='2027-01-01',
        min_battery_units=1,
        max_battery_units=5,
        min_solar_units=1,
        max_solar_units=10
    )


@pytest.fixture
def twoyear_scenario_spec():
    return SharedTestFixtures.create_default_scenario_spec(
        start_date='2026-01-01',
        end_date='2028-01-01',
        min_battery_units=1, 
        max_battery_units=5,
        min_solar_units=1, 
        max_solar_units=10
    )

def dummy_optimizer_results(start: pd.Timestamp, end: pd.Timestamp, freq):
    """Create sample optimizer results for testing."""
    index = pd.date_range(start=start, end=end - MIN_DT, freq=freq)
    n = len(index)

    results_df = pd.DataFrame({
        'P_batt': np.random.randn(n) * 2,
        'P_grid': np.random.randn(n) * 5,
        'P_subpanel': np.random.randn(n) * 3,
        'E': np.random.rand(n) * 13.5,
        'solar_post_curtailment': np.random.rand(n) * 5
    }, index=index)
    
    return OptimizerOutputs(
        results_df=results_df,
        status='optimal',
        sizing_results={'n_batt_blocks': 2, 'n_solar': 3}
    )


class TestBasicResultSummarizer:
    """Test the BasicResultSummarizer class."""


    def test_annual_financial_timeseries(self, twoyear_scenario_spec):
        scenario_spec = twoyear_scenario_spec
        design_inputs = scenario_spec.build_design_inputs()
        financial_inputs = scenario_spec.build_financial_model_inputs()

        site_data_index = design_inputs.site_data.index
        sample_optimizer_results = dummy_optimizer_results(start=scenario_spec.general_assumptions.start_date,
                                                           end=scenario_spec.general_assumptions.end_date,
                                                           freq=site_data_index.freq)
        """Test annual timeseries creation."""
        summarizer = BasicResultSummarizer()

        annual_df = summarizer._create_annual_financial_timeseries(
            optimization_results=sample_optimizer_results,
            tariff_model=design_inputs.tariff_model,
            financial_inputs=financial_inputs
        )

        assert len(annual_df) == scenario_spec.general_assumptions.study_years
        assert sorted(annual_df.columns.names) == sorted(['expense_type', 'category'])
        tmp = annual_df.copy()
        tmp.index = pd.DatetimeIndex(tmp.index)
        assert tmp.index.equals(tmp.resample('1YS').first().index)

        for product, product_cash_flows in financial_inputs.product_cash_flows.items():
            product_subset = annual_df.xs(product, level='category', axis=1)
            product_sizing_field = PRODUCT_TO_SIZING_OUTPUT_MAP[product]
            n_product_units = sample_optimizer_results.sizing_results[product_sizing_field]
            assert np.allclose(product_subset.values, product_cash_flows.unit_cash_flows * n_product_units)


    def test_summarizer(self, oneyear_scenario_spec):
        """Test basic summarization functionality."""
        scenario_spec = oneyear_scenario_spec
        design_inputs = scenario_spec.build_design_inputs()
        financial_inputs = scenario_spec.build_financial_model_inputs()

        summarizer = BasicResultSummarizer()

        site_data_index = design_inputs.site_data.index
        sample_optimizer_results = dummy_optimizer_results(start=site_data_index[0],
                                                           end=site_data_index[-1],
                                                           freq=site_data_index.freq)
        
        result = summarizer.summarize(
            sample_optimizer_results,
            design_inputs,
            financial_inputs,
            design_inputs.tariff_model
        )
        
        assert isinstance(result, ResultSummary)
        assert result.combined_timeseries.index.equals(sample_optimizer_results.results_df.index)
        assert not result.annual_nonfinancial_timeseries.empty
        assert not result.summary_stats.empty
        assert result.financial_summary is not None
        assert result.optimization_status == 'optimal'


@pytest.mark.slow
def test_scenario_dispatch_small_sweep(short_scenario_spec):
    scenario_spec = short_scenario_spec
    """Test scenario dispatch with a small sweep."""
    runner = SizingSweepScenarioRunner(scenario_spec=scenario_spec)

    runner.run_scenarios()
    optimizer_results = runner.get_optimizer_results()
    result_summaries = runner.get_result_summaries()
    n_battery_options = scenario_spec.design_spec.max_battery_units - scenario_spec.design_spec.min_battery_units + 1
    n_solar_options = scenario_spec.design_spec.max_solar_units - scenario_spec.design_spec.min_solar_units + 1

    assert len(result_summaries) == n_battery_options * n_solar_options
    for i, summary in enumerate(result_summaries):
        assert isinstance(summary, ResultSummary)
        assert summary.optimization_status == 'optimal'
        rd = relativedelta(summary.combined_timeseries.index[-1], summary.combined_timeseries.index[0])
        billing_months = rd.years * 12 + rd.months + int(rd.days > 0)
        billing_years = rd.years + int(rd.months > 0 or rd.days > 0)
        assert len(summary.billing_cycles) == billing_months
        assert len(summary.annual_financial_timeseries) == billing_years
        assert len(summary.annual_nonfinancial_timeseries) == billing_years

    print("Done")

@pytest.mark.slow
def test_topn_scenario_runner(short_scenario_spec):
    """Test TopNScenarioRunner initialization."""
    scenario_spec = short_scenario_spec
    n_closest = 3
    runner = TopNScenarioRunner(
        scenario_spec=scenario_spec,
        n_closest=n_closest
    )

    runner.run_scenarios()
    optimizer_results = runner.get_optimizer_results()
    result_summaries = runner.get_result_summaries()

    assert len(result_summaries) == n_closest  # 2 battery options * 2 solar options
    for i, summary in enumerate(result_summaries):
        assert isinstance(summary, ResultSummary)
        assert summary.optimization_status == 'optimal'
        rd = relativedelta(summary.combined_timeseries.index[-1], summary.combined_timeseries.index[0])
        billing_months = rd.years * 12 + rd.months + int(rd.days > 0)
        billing_years = rd.years + int(rd.months > 0 or rd.days > 0)
        assert len(summary.billing_cycles) == billing_months
        assert len(summary.annual_financial_timeseries) == billing_years
        assert len(summary.annual_nonfinancial_timeseries) == billing_years



if __name__ == "__main__":
    pytest.main([__file__])