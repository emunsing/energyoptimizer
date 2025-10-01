import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.energyoptimizer.batteryopt_interface import (
    DesignSpec, FinancialSpec, GeneralAssumptions, TariffSpec, ScenarioSpec,
    DesignInputs, FinancialModelInputs
)
from src.energyoptimizer.tariff.tariff_utils import TariffModel


class TestDesignSpec:
    """Test DesignSpec methods for building timeseries data."""
    
    def test_build_solar_timeseries_upload(self):
        """Test solar timeseries with uploaded data."""
        # Create test data
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        solar_data = pd.DataFrame(np.random.rand(24) * 5, index=time_index)
        
        # Create DesignSpec with uploaded data
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=solar_data
        )
        
        # Test building solar timeseries
        result = design_spec.build_solar_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'solar'
        pd.testing.assert_series_equal(result, solar_data.iloc[:, 0], check_names=False)
    
    def test_build_solar_timeseries_year1_degradation(self):
        """Test solar timeseries with year 1 data and degradation."""
        # Create test data for year 1
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        solar_data = pd.DataFrame(np.ones(24) * 5, index=time_index)
        
        # Create DesignSpec with year 1 data
        design_spec = DesignSpec(
            solar_data_source="year1",
            solar_data=solar_data,
            solar_first_year_degradation=0.1,  # 10% first year
            solar_subsequent_year_degradation=0.05  # 5% subsequent years
        )
        
        # Test with multi-year data
        multi_year_index = pd.date_range('2023-01-01', periods=48, freq='h')
        result = design_spec.build_solar_timeseries(multi_year_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 48
        # First 24 hours should have 10% degradation
        assert np.allclose(result.iloc[:24], 5 * 0.9, rtol=1e-10)
        # Next 24 hours should have additional 5% degradation
        assert np.allclose(result.iloc[24:], 5 * 0.9 * 0.95, rtol=1e-10)
    
    def test_build_solar_timeseries_no_data(self):
        """Test solar timeseries with no data (defaults to zeros)."""
        design_spec = DesignSpec()
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        
        result = design_spec.build_solar_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'solar'
        assert np.allclose(result, 0.0)
    
    def test_build_circuit_load_timeseries_upload(self):
        """Test circuit load timeseries with uploaded data."""
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        circuit_load_data = pd.DataFrame(np.random.rand(24) * 10, index=time_index)
        
        design_spec = DesignSpec(
            circuit_load_data_source="upload",
            circuit_load_data=circuit_load_data
        )
        
        result = design_spec.build_circuit_load_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'circuit_load'
        pd.testing.assert_series_equal(result, circuit_load_data.iloc[:, 0])
    
    def test_build_circuit_load_timeseries_no_data(self):
        """Test circuit load timeseries with no data (defaults to zeros)."""
        design_spec = DesignSpec()
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        
        result = design_spec.build_circuit_load_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'circuit_load'
        assert np.allclose(result, 0.0)
    
    def test_build_non_circuit_load_timeseries_upload(self):
        """Test non-circuit load timeseries with uploaded data."""
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        non_circuit_load_data = pd.DataFrame(np.random.rand(24) * 5, index=time_index)
        
        design_spec = DesignSpec(
            non_circuit_load_data_source="upload",
            non_circuit_load_data=non_circuit_load_data
        )
        
        result = design_spec.build_non_circuit_load_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'non_circuit_load'
        pd.testing.assert_series_equal(result, non_circuit_load_data.iloc[:, 0])
    
    def test_build_non_circuit_load_timeseries_flat(self):
        """Test non-circuit load timeseries with flat profile."""
        design_spec = DesignSpec(
            non_circuit_load_data_source="flat",
            facility_sqft=10000,  # 10,000 sq ft
            annual_eui=20  # 20 kWh/sqft-yr
        )
        
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        result = design_spec.build_non_circuit_load_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'non_circuit_load'
        
        # Calculate expected hourly energy
        annual_energy = 10000 * 20  # kWh
        hourly_energy = annual_energy / 8760  # kWh/hr
        assert np.allclose(result, hourly_energy)
    
    def test_build_non_circuit_load_timeseries_no_data(self):
        """Test non-circuit load timeseries with no data (defaults to zeros)."""
        design_spec = DesignSpec()
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        
        result = design_spec.build_non_circuit_load_timeseries(time_index)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 24
        assert result.name == 'non_circuit_load'
        assert np.allclose(result, 0.0)


class TestTariffSpec:
    """Test TariffSpec build_tariff method."""
    
    def test_build_tariff(self):
        """Test building tariff from TariffSpec."""
        tariff_spec = TariffSpec(
            rate_code="test_rate",
            annual_rate_escalator=0.03
        )
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        
        # This will fail if tariffs.yaml doesn't exist or doesn't contain test_rate
        # For now, we'll test that the method can be called
        try:
            tariff_model = tariff_spec.build_tariff(start_date, end_date)
            assert isinstance(tariff_model, TariffModel)
        except (FileNotFoundError, KeyError):
            # Expected if tariffs.yaml doesn't exist or test_rate not found
            pytest.skip("tariffs.yaml not found or test_rate not available")


class TestScenarioSpec:
    """Test ScenarioSpec methods for building complete inputs."""
    
    def test_build_tariff(self):
        """Test building tariff from ScenarioSpec."""
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            study_years=1
        )
        tariff_spec = TariffSpec(rate_code="test_rate")
        design_spec = DesignSpec()
        financial_spec = FinancialSpec()
        
        scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
        
        try:
            tariff_model = scenario_spec.build_tariff()
            assert isinstance(tariff_model, TariffModel)
        except (FileNotFoundError, KeyError):
            pytest.skip("tariffs.yaml not found or test_rate not available")
    
    def test_build_design_inputs(self):
        """Test building design inputs from ScenarioSpec."""
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            study_years=1,
            study_resolution='h'
        )
        
        # Create test data
        time_index = pd.date_range('2023-01-01', periods=24, freq='h')
        solar_data = pd.DataFrame(np.ones(24) * 5, index=time_index)
        circuit_load_data = pd.DataFrame(np.ones(24) * 10, index=time_index)
        non_circuit_load_data = pd.DataFrame(np.ones(24) * 3, index=time_index)
        
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=solar_data,
            circuit_load_data_source="upload",
            circuit_load_data=circuit_load_data,
            non_circuit_load_data_source="upload",
            non_circuit_load_data=non_circuit_load_data
        )
        
        tariff_spec = TariffSpec(rate_code="test_rate")
        financial_spec = FinancialSpec()
        
        scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
        
        try:
            design_inputs = scenario_spec.build_design_inputs()
            assert isinstance(design_inputs, DesignInputs)
            assert isinstance(design_inputs.site_data, pd.DataFrame)
            assert 'solar' in design_inputs.site_data.columns
            assert 'circuit_load' in design_inputs.site_data.columns
            assert 'non_circuit_load' in design_inputs.site_data.columns
            assert len(design_inputs.site_data) == 24
        except (FileNotFoundError, KeyError):
            pytest.skip("tariffs.yaml not found or test_rate not available")
    
    def test_build_financial_model_inputs(self):
        """Test building financial model inputs from ScenarioSpec."""
        general_assumptions = GeneralAssumptions()
        design_spec = DesignSpec()
        tariff_spec = TariffSpec(rate_code="test_rate")
        financial_spec = FinancialSpec(
            study_years=15,
            discount_rate=0.08,
            solar_capital_cost_per_unit=2500.0,
            battery_capital_cost_per_unit=800.0
        )
        
        scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
        
        financial_inputs = scenario_spec.build_financial_model_inputs()
        
        assert isinstance(financial_inputs, FinancialModelInputs)
        assert financial_inputs.study_years == 15
        assert financial_inputs.discount_rate == 0.08
        assert financial_inputs.solar_capital_cost_per_unit == 2500.0
        assert financial_inputs.battery_capital_cost_per_unit == 800.0


class TestIntegration:
    """Test integration between interface classes and OptimizationRunnerInputs."""
    
    def test_scenario_spec_to_optimization_runner_inputs(self):
        """Test that ScenarioSpec can create inputs for OptimizationRunnerInputs."""
        from src.energyoptimizer.optimization_runner import OptimizationRunnerInputs, OptimizationType
        
        # Create a complete scenario
        general_assumptions = GeneralAssumptions(
            start_date='2023-01-01',
            study_years=1,
            optimization_type='self_consumption'
        )
        
        design_spec = DesignSpec()
        tariff_spec = TariffSpec(rate_code="test_rate")
        financial_spec = FinancialSpec()
        
        scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
        
        try:
            # Build inputs
            design_inputs = scenario_spec.build_design_inputs()
            financial_inputs = scenario_spec.build_financial_model_inputs()
            
            # Create OptimizationRunnerInputs
            runner_inputs = OptimizationRunnerInputs(
                optimization_type=OptimizationType.SELF_CONSUMPTION,
                optimization_start=general_assumptions.start_date,
                optimization_end=general_assumptions.end_date,
                design_inputs=design_inputs,
                financial_model_inputs=financial_inputs
            )
            
            assert isinstance(runner_inputs, OptimizationRunnerInputs)
            assert runner_inputs.optimization_type == OptimizationType.SELF_CONSUMPTION
            assert isinstance(runner_inputs.design_inputs, DesignInputs)
            assert isinstance(runner_inputs.financial_model_inputs, FinancialModelInputs)
        except (FileNotFoundError, KeyError):
            pytest.skip("tariffs.yaml not found or test_rate not available")
