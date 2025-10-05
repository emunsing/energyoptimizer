import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

from src.energyoptimizer.scenario_runner import (
    closest_n_elements, ResultSummary, BasicResultSummarizer, 
    ScenarioRunner, SizingSweepScenarioRunner, TopNScenarioRunner
)
from src.energyoptimizer.batteryopt_interface import (
    ScenarioSpec, GeneralAssumptions, DesignSpec, TariffSpec, FinancialSpec,
    DesignInputs, FinancialModelInputs, OptimizationType, OptimizationClock,
    OptimizationRunnerInputs, ProductCashFlows, PRODUCT_TO_SIZING_OUTPUT_MAP
)
from src.energyoptimizer.batteryopt_utils import MIN_DT
from src.energyoptimizer.optimizers import OptimizerOutputs
from src.energyoptimizer.tariff.tariff_utils import TariffModel
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
        financial_spec = FinancialSpec(study_years=1)
        
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
        end_date='2026-06-01',
        min_battery_units=1, 
        max_battery_units=2,
        min_solar_units=1, 
        max_solar_units=3
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


class TestScenarioRunner:
    """Test the base ScenarioRunner class."""
    
    def test_scenario_runner_initialization(self, short_scenario_spec):
        """Test ScenarioRunner initialization."""
        # Can't instantiate abstract class directly, so test via subclass
        runner = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        assert runner.scenario_spec.general_assumptions == short_scenario_spec.general_assumptions
        assert runner.scenario_spec.design_spec == short_scenario_spec.design_spec
        assert runner.scenario_spec.tariff_spec == short_scenario_spec.tariff_spec
        assert runner.scenario_spec.financial_spec == short_scenario_spec.financial_spec
        assert runner.design_inputs is not None
        assert runner.financial_inputs is not None
        assert runner.tariff_model is not None
        assert runner.optimizer_results == []
        assert runner.result_summaries == []


class TestSizingSweepScenarioRunner:
    """Test the SizingSweepScenarioRunner class."""
    
    def test_sizing_sweep_initialization(self, short_scenario_spec):
        """Test SizingSweepScenarioRunner initialization."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        assert runner.n_batt_min == 1
        assert runner.n_batt_max == 2
        assert runner.solar_min == 1
        assert runner.solar_max == 2
    
    def test_create_modified_scenario_spec(self, short_scenario_spec):
        """Test creation of modified scenario specs."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        modified_spec = runner._create_modified_scenario_spec(n_batt=2, solar_size=3)
        
        assert modified_spec.design_spec.min_battery_units == 2
        assert modified_spec.design_spec.max_battery_units == 2
        assert modified_spec.design_spec.min_solar_units == 3
        assert modified_spec.design_spec.max_solar_units == 3
    
    @pytest.mark.slow
    def test_scenario_dispatch_small_sweep(self, short_scenario_spec):
        """Test scenario dispatch with a small sweep."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=1,  # Only 1 battery option
            solar_min=1, solar_max=1     # Only 1 solar option
        )
        
        # Mock the OptimizationRunner to avoid actual optimization
        with pytest.Mock() as mock_runner:
            mock_result = OptimizerOutputs(
                results_df=pd.DataFrame({'P_batt': [0], 'P_grid': [1]}, 
                                       index=pd.date_range('2023-01-01', periods=1, freq='H')),
                status='optimal'
            )
            mock_runner.run_optimization.return_value = mock_result
            
            # This would normally call the real optimizer, but we're mocking it
            # For now, just test that the structure is correct
            assert runner.scenario_spec is not None


class TestTopNScenarioRunner:
    """Test the TopNScenarioRunner class."""
    
    def test_topn_initialization(self, medium_scenario_spec):
        """Test TopNScenarioRunner initialization."""
        runner = TopNScenarioRunner(
            general_assumptions=medium_scenario_spec.general_assumptions,
            design_spec=medium_scenario_spec.design_spec,
            tariff_spec=medium_scenario_spec.tariff_spec,
            financial_spec=medium_scenario_spec.financial_spec,
            n_closest=5
        )
        
        assert runner.n_closest == 5
        assert runner.endogenous_result is None
    
    def test_create_modified_scenario_spec(self, medium_scenario_spec):
        """Test creation of modified scenario specs."""
        runner = TopNScenarioRunner(
            general_assumptions=medium_scenario_spec.general_assumptions,
            design_spec=medium_scenario_spec.design_spec,
            tariff_spec=medium_scenario_spec.tariff_spec,
            financial_spec=medium_scenario_spec.financial_spec,
            n_closest=3
        )
        
        modified_spec = runner._create_modified_scenario_spec(n_batt=3, solar_size=7)
        
        assert modified_spec.design_spec.min_battery_units == 3
        assert modified_spec.design_spec.max_battery_units == 3
        assert modified_spec.design_spec.min_solar_units == 7
        assert modified_spec.design_spec.max_solar_units == 7


class TestIntegration:
    """Integration tests for the scenario runner system."""
    
    def test_result_summarizer_protocol(self):
        """Test that BasicResultSummarizer implements the ResultSummarizer protocol."""
        summarizer = BasicResultSummarizer()
        
        # Should have the required summarize method
        assert hasattr(summarizer, 'summarize')
        assert callable(getattr(summarizer, 'summarize'))
    
    def test_scenario_runner_composability(self):
        """Test that scenario runners can use different result summarizers."""
        # Create a custom summarizer
        class CustomSummarizer:
            def summarize(self, optimizer_results, design_inputs, financial_inputs, tariff_model):
                return ResultSummary(optimization_status='custom')
        
        # This should work with any scenario runner
        custom_summarizer = CustomSummarizer()
        
        # The interface should be compatible
        assert hasattr(custom_summarizer, 'summarize')
        assert callable(getattr(custom_summarizer, 'summarize'))
    
    def test_parallelization_support(self, short_scenario_spec):
        """Test that parallelization can be enabled/disabled."""
        # Test with parallelization disabled
        runner_seq = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=1,
            solar_min=1, solar_max=1,
            parallelize=False
        )
        
        # Test with parallelization enabled
        runner_par = SizingSweepScenarioRunner(
            general_assumptions=short_scenario_spec.general_assumptions,
            design_spec=short_scenario_spec.design_spec,
            tariff_spec=short_scenario_spec.tariff_spec,
            financial_spec=short_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=1,
            solar_min=1, solar_max=1,
            parallelize=True,
            n_processes=2
        )
        
        assert runner_seq.parallelize == False
        assert runner_par.parallelize == True
        assert runner_par.n_processes == 2


if __name__ == "__main__":
    pytest.main([__file__])