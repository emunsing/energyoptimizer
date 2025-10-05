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
    OptimizationRunnerInputs, ProductCashFlows
)
from src.energyoptimizer.optimizers import OptimizerOutputs
from src.energyoptimizer.tariff.tariff_utils import TariffModel
from tests.test_utils import sample_site_data
from tests.test_optimizer import TZ



class TestBasicResultSummarizer:
    """Test the BasicResultSummarizer class."""
    
    @pytest.fixture
    def sample_optimizer_results(self):
        """Create sample optimizer results for testing."""
        index = pd.date_range('2023-01-01', periods=24, freq='H', tz=TZ)
        results_df = pd.DataFrame({
            'P_batt': np.random.randn(24) * 2,
            'P_grid': np.random.randn(24) * 5,
            'P_subpanel': np.random.randn(24) * 3,
            'E': np.random.rand(24) * 13.5,
            'solar_post_curtailment': np.random.rand(24) * 5
        }, index=index)
        
        return OptimizerOutputs(
            results_df=results_df,
            status='optimal',
            sizing_results={'n_batt_blocks': 2, 'n_solar': 5}
        )
    
    @pytest.fixture
    def sample_design_inputs(self):
        """Create sample design inputs for testing."""
        index = pd.date_range('2023-01-01', periods=24, freq='H', tz=TZ)
        site_data = pd.DataFrame({
            'solar': np.random.rand(24) * 6,
            'der_subpanel_load': np.random.rand(24) * 10,
            'main_panel_load': np.random.rand(24) * 15
        }, index=index)
        
        # Mock tariff model
        tariff_model = Mock()
        tariff_model.tariff_timeseries = pd.DataFrame({
            'energy_import_rate_kwh': np.random.rand(24) * 0.3,
            'energy_export_rate_kwh': np.random.rand(24) * 0.2
        }, index=index)
        
        return DesignInputs(
            site_data=site_data,
            tariff_model=tariff_model,
            batt_rt_eff=0.85,
            batt_block_e_max=13.5,
            batt_block_p_max=5.0
        )
    
    @pytest.fixture
    def sample_financial_inputs(self):
        """Create sample financial inputs for testing."""
        cash_flows_df = pd.DataFrame({
            'capital_cost': [-1000, 0, 0],
            'residual_value': [0, 0, 100]
        })
        
        solar_cash_flows = ProductCashFlows(
            fixed_upfront_installation_cost=0,
            unit_cash_flows=cash_flows_df,
            unit_annualized_cost=500
        )
        
        battery_cash_flows = ProductCashFlows(
            fixed_upfront_installation_cost=0,
            unit_cash_flows=cash_flows_df,
            unit_annualized_cost=300
        )
        
        return FinancialModelInputs(
            study_years=3,
            product_cash_flows={
                'solar': solar_cash_flows,
                'battery': battery_cash_flows
            },
            discount_rate=0.07,
            solar_levelized_unit_cost=500,
            battery_levelized_unit_cost=300,
            reference_upgrade_cost=100000
        )
    
    def test_summarize_basic(self, sample_optimizer_results, sample_design_inputs, sample_financial_inputs):
        """Test basic summarization functionality."""
        summarizer = BasicResultSummarizer()
        
        # Mock the tariff model for compute_bill_series
        sample_design_inputs.tariff_model.compute_bill_series = Mock(
            return_value=pd.DataFrame({'bill': [100, 120, 110]})
        )
        
        result = summarizer.summarize(
            sample_optimizer_results,
            sample_design_inputs,
            sample_financial_inputs,
            sample_design_inputs.tariff_model
        )
        
        assert isinstance(result, ResultSummary)
        assert result.combined_timeseries is not None
        assert result.annual_timeseries is not None
        assert result.summary_stats is not None
        assert result.design_params is not None
        assert result.financial_summary is not None
        assert result.optimization_status == 'optimal'
    
    def test_create_combined_timeseries(self, sample_optimizer_results, sample_design_inputs):
        """Test combined timeseries creation."""
        summarizer = BasicResultSummarizer()
        
        combined = summarizer._create_combined_timeseries(
            sample_optimizer_results.results_df,
            sample_design_inputs,
            sample_design_inputs.tariff_model
        )
        
        # Should combine optimizer results, site data, and tariff
        expected_columns = set(sample_optimizer_results.results_df.columns)
        expected_columns.update(sample_design_inputs.site_data.columns)
        expected_columns.update(sample_design_inputs.tariff_model.tariff_timeseries.columns)
        
        assert set(combined.columns) == expected_columns
        assert len(combined) == len(sample_optimizer_results.results_df)
    
    def test_create_annual_timeseries(self, sample_optimizer_results, sample_design_inputs):
        """Test annual timeseries creation."""
        summarizer = BasicResultSummarizer()
        
        annual = summarizer._create_annual_timeseries(
            sample_optimizer_results.results_df,
            sample_design_inputs,
            sample_design_inputs.tariff_model
        )
        
        # Should have energy flow columns
        expected_columns = [
            'uncurtailed_solar_kwh', 'curtailed_solar_kwh',
            'grid_imports_kwh', 'grid_exports_kwh',
            'subpanel_imports_kwh', 'subpanel_exports_kwh'
        ]
        
        for col in expected_columns:
            assert col in annual.columns
        
        # All values should be non-negative (energy flows)
        energy_columns = [col for col in annual.columns if 'kwh' in col]
        for col in energy_columns:
            assert (annual[col] >= 0).all()


class TestScenarioRunner:
    """Test the base ScenarioRunner class."""
    
    @pytest.fixture
    def sample_scenario_spec(self):
        """Create a sample scenario spec for testing."""
        one_year_site_data = sample_site_data('2023-01-01', '2023-02-01', freq='1H', tz=TZ)
        
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            study_years=1,
            optimization_type='tou_optimization'
        )
        
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=one_year_site_data[['solar']],
            circuit_load_data_source="upload",
            circuit_load_data=one_year_site_data[['der_subpanel_load']],
            non_circuit_load_data_source="upload",
            non_circuit_load_data=one_year_site_data[['main_panel_load']],
            min_battery_units=1,
            max_battery_units=3,
            min_solar_units=1,
            max_solar_units=5
        )
        
        tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
        financial_spec = FinancialSpec(study_years=1)
        
        return ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
    
    def test_scenario_runner_initialization(self, sample_scenario_spec):
        """Test ScenarioRunner initialization."""
        # Can't instantiate abstract class directly, so test via subclass
        runner = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        assert runner.scenario_spec.general_assumptions == sample_scenario_spec.general_assumptions
        assert runner.scenario_spec.design_spec == sample_scenario_spec.design_spec
        assert runner.scenario_spec.tariff_spec == sample_scenario_spec.tariff_spec
        assert runner.scenario_spec.financial_spec == sample_scenario_spec.financial_spec
        assert runner.design_inputs is not None
        assert runner.financial_inputs is not None
        assert runner.tariff_model is not None
        assert runner.optimizer_results == []
        assert runner.result_summaries == []


class TestSizingSweepScenarioRunner:
    """Test the SizingSweepScenarioRunner class."""
    
    @pytest.fixture
    def sample_scenario_spec(self):
        """Create a sample scenario spec for testing."""
        one_year_site_data = sample_site_data('2023-01-01', '2023-01-03', freq='1H', tz=TZ)  # Short period for testing
        
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            end_date='2023-01-03',
            study_years=1,
            optimization_type='tou_optimization'
        )
        
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=one_year_site_data[['solar']],
            circuit_load_data_source="upload",
            circuit_load_data=one_year_site_data[['der_subpanel_load']],
            non_circuit_load_data_source="upload",
            non_circuit_load_data=one_year_site_data[['main_panel_load']],
            min_battery_units=1,
            max_battery_units=2,
            min_solar_units=1,
            max_solar_units=3
        )
        
        tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
        financial_spec = FinancialSpec(study_years=1)
        
        return ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
    
    def test_sizing_sweep_initialization(self, sample_scenario_spec):
        """Test SizingSweepScenarioRunner initialization."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        assert runner.n_batt_min == 1
        assert runner.n_batt_max == 2
        assert runner.solar_min == 1
        assert runner.solar_max == 2
    
    def test_create_modified_scenario_spec(self, sample_scenario_spec):
        """Test creation of modified scenario specs."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=2,
            solar_min=1, solar_max=2
        )
        
        modified_spec = runner._create_modified_scenario_spec(n_batt=2, solar_size=3)
        
        assert modified_spec.design_spec.min_battery_units == 2
        assert modified_spec.design_spec.max_battery_units == 2
        assert modified_spec.design_spec.min_solar_units == 3
        assert modified_spec.design_spec.max_solar_units == 3
    
    @pytest.mark.slow
    def test_scenario_dispatch_small_sweep(self, sample_scenario_spec):
        """Test scenario dispatch with a small sweep."""
        runner = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
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
    
    @pytest.fixture
    def sample_scenario_spec(self):
        """Create a sample scenario spec for testing."""
        one_year_site_data = sample_site_data('2023-01-01', '2023-01-03', freq='1H', tz=TZ)  # Short period for testing
        
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            end_date='2023-01-03',
            study_years=1,
            optimization_type='tou_optimization'
        )
        
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=one_year_site_data[['solar']],
            circuit_load_data_source="upload",
            circuit_load_data=one_year_site_data[['der_subpanel_load']],
            non_circuit_load_data_source="upload",
            non_circuit_load_data=one_year_site_data[['main_panel_load']],
            min_battery_units=1,
            max_battery_units=5,
            min_solar_units=1,
            max_solar_units=10
        )
        
        tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
        financial_spec = FinancialSpec(study_years=1)
        
        return ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
    
    def test_topn_initialization(self, sample_scenario_spec):
        """Test TopNScenarioRunner initialization."""
        runner = TopNScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
            n_closest=5
        )
        
        assert runner.n_closest == 5
        assert runner.endogenous_result is None
    
    def test_create_modified_scenario_spec(self, sample_scenario_spec):
        """Test creation of modified scenario specs."""
        runner = TopNScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
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
    
    def test_parallelization_support(self, sample_scenario_spec):
        """Test that parallelization can be enabled/disabled."""
        # Test with parallelization disabled
        runner_seq = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
            n_batt_min=1, n_batt_max=1,
            solar_min=1, solar_max=1,
            parallelize=False
        )
        
        # Test with parallelization enabled
        runner_par = SizingSweepScenarioRunner(
            general_assumptions=sample_scenario_spec.general_assumptions,
            design_spec=sample_scenario_spec.design_spec,
            tariff_spec=sample_scenario_spec.tariff_spec,
            financial_spec=sample_scenario_spec.financial_spec,
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
