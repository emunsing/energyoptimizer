import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

from src.energyoptimizer.batteryopt_interface import (
    DesignSpec, FinancialSpec, GeneralAssumptions, TariffSpec, ScenarioSpec,
    DesignInputs, FinancialModelInputs
)
from src.energyoptimizer.tariff.tariff_utils import TariffModel
from .test_utils import sample_site_data
from src.energyoptimizer.optimization_runner import OptimizationRunnerInputs, OptimizationType

TZ = 'US/Pacific'


class TestDesignSpec:
    """Test DesignSpec methods for building timeseries data."""
    time_index = pd.date_range('2025-01-01', '2044-12-31 23:59', tz=TZ, freq='1H')
    one_year_site_data = sample_site_data('2023-01-01', '2024-01-01', freq='1H', tz=TZ)
    
    def test_build_solar_timeseries_upload(self):
        # Create DesignSpec with uploaded data
        design_spec = DesignSpec(
            solar_data_source="upload",
            solar_data=self.one_year_site_data
        )
        result = design_spec.build_solar_timeseries(self.time_index)
        assert result.index.equals(self.time_index)
        assert result.name == 'solar'
        assert result.notnull().all()

        # Confirm that solar decays over time
        annual_max = result.groupby(result.index.year).max()
        assert annual_max.iloc[1] == annual_max.values[0] * (1 - design_spec.solar_first_year_degradation)
        assert annual_max.iloc[-1] < annual_max.values[0] * 0.85
    
    def test_build_circuit_load_timeseries_upload(self):
        design_spec = DesignSpec(
            circuit_load_data_source="upload",
            circuit_load_data=self.one_year_site_data
        )
        result = design_spec.build_circuit_load_timeseries(self.time_index)
        assert result.index.equals(self.time_index)
        assert result.name == 'der_subpanel_load'
        assert result.notnull().all()

        # Confirm that results are roughly constant, allowing for leap years
        annual_total = result.groupby(result.index.year).sum()
        annual_total_deviation = annual_total / annual_total.mean() - 1
        assert all(annual_total_deviation.abs() < 0.01)

    def test_build_non_circuit_load_timeseries_upload(self):
        design_spec = DesignSpec(
            non_circuit_load_data_source="upload",
            non_circuit_load_data=self.one_year_site_data
        )
        result = design_spec.build_non_circuit_load_timeseries(self.time_index)
        assert result.index.equals(self.time_index)
        assert result.name == 'main_panel_load'
        assert result.notnull().all()

        # Confirm that results are roughly constant, allowing for leap years
        annual_total = result.groupby(result.index.year).sum()
        annual_total_deviation = annual_total / annual_total.mean() - 1
        assert all(annual_total_deviation.abs() < 0.01)
    
    def test_build_non_circuit_load_timeseries_flat(self):
        """Test non-circuit load timeseries with flat profile."""
        annual_eui = 6 # kWh/sqft/yr
        building_size = 10e3
        design_spec = DesignSpec(
            non_circuit_load_data_source="flat",
            facility_sqft=building_size,
            annual_eui=annual_eui,
        )
        result = design_spec.build_non_circuit_load_timeseries(self.time_index)
        assert result.index.equals(self.time_index)
        assert result.name == 'main_panel_load'
        assert result.notnull().all()

        # Calculate expected hourly energy
        hourly_energy = building_size * annual_eui / 8760
        assert np.allclose(result, hourly_energy)



class TestScenarioSpec:
    """Test ScenarioSpec methods for building complete inputs."""
    one_year_site_data = sample_site_data('2023-01-01', '2024-01-01', freq='1H', tz=TZ)

    default_tariff_spec = TariffSpec(rate_code="PGE_B_19_R")
    default_design_spec = DesignSpec(solar_data_source="upload",
                                     solar_data=one_year_site_data['solar'],
                                     circuit_load_data_source="upload",
                                     circuit_load_data=one_year_site_data['der_subpanel_load'],
                                     non_circuit_load_data_source="upload",
                                     non_circuit_load_data=one_year_site_data['main_panel_load']
                                     )
    default_financial_spec = FinancialSpec()
    default_general_assumptions = GeneralAssumptions(start_date='2026-01-01',
                                                     study_years=25
                                                     )
    default_scenario_spec = ScenarioSpec(
            general_assumptions=default_general_assumptions,
            design_spec=default_design_spec,
            tariff_spec=default_tariff_spec,
            financial_spec=default_financial_spec
        )
    
    def test_build_tariff(self):
        """Test building tariff from ScenarioSpec."""
        tariff_model = self.default_scenario_spec.build_tariff()
        assert isinstance(tariff_model, TariffModel)
        # Test that tariff components are built
        assert hasattr(tariff_model, 'tariff_timeseries')
        assert hasattr(tariff_model, 'demand_charge_categorical_dataframe')
        assert hasattr(tariff_model, 'demand_charge_price_map')
        assert hasattr(tariff_model, 'billing_cycles')


    def test_build_design_inputs(self):
        design_inputs = self.default_scenario_spec.build_design_inputs()
        assert isinstance(design_inputs, DesignInputs)
        site_data = design_inputs.site_data
        assert isinstance(site_data, pd.DataFrame)
        assert sorted(['solar', 'der_subpanel_load', 'main_panel_load']) == sorted(site_data.columns)
        assert site_data.index.freq == self.default_general_assumptions.study_resolution
        site_data_duration_years = relativedelta(site_data.index[-1] + site_data.index.freq, site_data.index[0]).years
        assert site_data_duration_years == self.default_general_assumptions.study_years

    def test_build_financial_model_inputs(self):
        financial_inputs = self.default_scenario_spec.build_financial_model_inputs()
        
        assert isinstance(financial_inputs, FinancialModelInputs)
        assert financial_inputs.study_years == 15
        assert financial_inputs.discount_rate == 0.08
        assert financial_inputs.solar_capital_cost_per_unit == 2500.0
        assert financial_inputs.battery_capital_cost_per_unit == 800.0


    def test_optimization_runner_inputs(self):
        design_inputs = self.default_scenario_spec.build_design_inputs()
        financial_inputs = self.default_scenario_spec.build_financial_model_inputs()

        # Create OptimizationRunnerInputs
        runner_inputs = OptimizationRunnerInputs(
            optimization_type=OptimizationType.SELF_CONSUMPTION,
            optimization_start=self.default_general_assumptions.start_date,
            optimization_end=self.default_general_assumptions.end_date,
            design_inputs=design_inputs,
            financial_model_inputs=financial_inputs
        )

        assert isinstance(runner_inputs, OptimizationRunnerInputs)

