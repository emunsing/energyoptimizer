from orange_constants import EV_PARAMETERS, CHARGE_SCHEDULE
import pandas as pd
import numpy as np
import os
import pathlib
import time
import attrs
from energyoptimizer.batteryopt_interface import DesignSpec, GeneralAssumptions, FinancialSpec, TariffSpec, ScenarioSpec
from energyoptimizer.scenario_runner import SizingSweepScenarioRunner, TopNScenarioRunner
from energyoptimizer.solar.solar_utils import get_or_cache_weather_data

DATA_CACHE_DIR = pathlib.Path(os.environ['SOLAR_CACHE_DIR'])# Ensure cache directory exists
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True) 

BASE_GENERAL_ASSUMPTIONS = GeneralAssumptions(
    start_date='2026-01-01',
    study_years=20,
    optimization_type='tou_endogenous_sizing',
    study_resolution='1h',
    optimization_clock='1YS',  # TODO: Confirm that this changes the clock
    optimization_clock_horizon=pd.DateOffset(months=12),
)   

BASE_DESIGN_SPEC = DesignSpec(
        min_battery_units=0,
        max_battery_units=10,
        min_solar_units=0,
        max_solar_units=20,
        # Battery spec
        battery_unit_size=13.5,  # kWh
        battery_unit_power=5,  # kW
        battery_rt_eff=0.9025,
        backup_reserve=0.1,
        # Circuit limits
        der_subpanel_import_kw_limit=100,  # kW
        der_subpanel_export_kw_limit=-100,  # kW
        site_import_limit=100,  # kW
        site_export_limit=-100,  # kW
        # Solar spec
        solar_data_source="upload",
        solar_data=None,
        solar_first_year_degradation=0.015,
        solar_subsequent_year_degradation=0.008,
        # Circuit load data
        circuit_load_data_source="upload",
        circuit_load_data=None,
        # Non-circuit load data
        non_circuit_load_data_source="flat",  # "upload" or "flat"
        facility_sqft=0,
        annual_eui=0,  # kWh/sqft/yr
        non_circuit_load_data=None,
    )

BASE_FINANCIAL_SPEC = FinancialSpec(study_years = 20,
                                    discount_rate = 0.06,

                                    # Solar costs
                                    solar_capital_cost_per_unit = 3500,
                                    solar_installationrepair_labor_cost_per_unit = 0,
                                    solar_fixed_upfront_installation_cost = 5000,
                                    solar_lifetime = 25,
                                    solar_residual_value = 0.25,
                                    solar_annual_inflation_rate = 0,

                                    # Battery costs
                                    battery_capital_cost_per_unit = 9500,
                                    battery_installation_cost_per_unit = 0 ,
                                    battery_fixed_upfront_installation_cost = 3000,
                                    battery_lifetime = 10,
                                    battery_residual_value = 0.0,
                                    battery_annual_inflation_rate = -0.03,

                                    # Financial incentives
                                    itc_rate = 0.3,
                                    itc_applies_to_replacement = False,
                                    apply_inflation_to_labor = False,

                                    # Other costs
                                    closing_costs_rate = 0.0,
                                    # These costs are not per-unit, and so do not fit well into the
                                    interconnection_allowance = 0.0,
                                    design_cost = 0.0,
                                    reference_upgrade_cost = 0.0,)

BASE_TARIFF_SPEC = TariffSpec(rate_code="PGE_E_ELEC",
                             annual_rate_escalator=0.05,
                             )

def run_sizing_optimization(load_data_series: pd.Series, 
                        latitude: float=37.774929, 
                        longitude: float=-122.419418
                        ):
    load_data_df = pd.DataFrame.from_dict({'der_subpanel_load': load_data_series}, orient='columns')

    solar_data = get_or_cache_weather_data(DATA_CACHE_DIR, latitude=latitude, longitude=longitude, 
                                            start_yr=2020, end_yr=2020, 
                                            timezone='America/Los_Angeles')

    general_assumptions = BASE_GENERAL_ASSUMPTIONS
    design_spec = attrs.evolve(BASE_DESIGN_SPEC, solar_data=solar_data, circuit_load_data=load_data_df)
    financial_spec = attrs.evolve(BASE_FINANCIAL_SPEC, study_years=general_assumptions.study_years)

    tariff_spec = BASE_TARIFF_SPEC

    runner = TopNScenarioRunner(
        general_assumptions=general_assumptions,
        design_spec=design_spec,
        tariff_spec=tariff_spec,
        financial_spec=financial_spec,
        n_closest=3,
        parallelize=False,
        n_jobs=6,
    )
    start_sim_at = time.time()
    runner.run_scenarios()
    end_sim_at = time.time()
    print(f"Simulation completed in {end_sim_at - start_sim_at:.2f} seconds")
    return runner.get_result_summaries()


"""
These classes are used to expose specific controls from the web interface to the user.
These will override the base 
"""

class GeneralAssumptionsWebInputs:
    study_years: int

class DesignSpecWebInputs:
    load_data_series: pd.Series
    min_battery_units: int
    max_battery_units: int  # Note: For the consumer sizing optimization, always set min_battery_units = max_battery_units = n_battery_units
    min_solar_units: int
    max_solar_units: int  # Note: For the consumer sizing optimization, always set min_solar_units = max_solar_units = solar_size_kw

class TariffSpecWebInputs:
    pass

class FinancialSpecWebInputs:
    discount_rate: float

class ConsumerWebInputs:
    general_assumptions_web_inputs: GeneralAssumptionsWebInputs
    design_spec_web_inputs: DesignSpecWebInputs
    tariff_spec_web_inputs: TariffSpecWebInputs
    financial_spec_web_inputs: FinancialSpecWebInputs

def run_single_optimization(consumer_web_inputs: ConsumerWebInputs,
                            ):
    """
    This should be able to override the base inputs with the fields wired up in the web interface. This should be flexible.
    """
    general_assumptions = attrs.evolve(BASE_GENERAL_ASSUMPTIONS, **consumer_web_inputs.general_assumptions_web_inputs)
    design_spec = attrs.evolve(BASE_DESIGN_SPEC, **consumer_web_inputs.design_spec_web_inputs)
    financial_spec = attrs.evolve(BASE_FINANCIAL_SPEC, **consumer_web_inputs.financial_spec_web_inputs)
    tariff_spec = attrs.evolve(BASE_TARIFF_SPEC, **consumer_web_inputs.tariff_spec_web_inputs)

    scenario = ScenarioSpec(
        general_assumptions=general_assumptions,
        design_spec=design_spec,
        tariff_spec=tariff_spec,
        financial_spec=financial_spec,
    )

    optimizer_runner_inputs = scenario.get_optimizer_runner_inputs()

    optimizer_runner = OptimizationRunner(optimizer_runner_inputs)
    optimizer_runner.run_optimization()
    return optimizer_runner.get_result_summaries()