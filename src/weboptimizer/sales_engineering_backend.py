from energyoptimizer.optimizers import OptimizerOutputs, WrappedOptimizerOutputs
import pathlib
import pandas as pd
import numpy as np
import datetime
import pickle
import time
import attrs
from attrs import asdict
from energyoptimizer.batteryopt_interface import DesignSpec, GeneralAssumptions, FinancialSpec, TariffSpec, ScenarioSpec
from energyoptimizer.tariff.tariff_utils import TariffModel
from energyoptimizer.scenario_runner import SizingSweepScenarioRunner, TopNScenarioRunner, BasicResultSummarizer
from pandas.tseries.frequencies import to_offset

TIMEZONE = 'US/Pacific'

@attrs.define
class ScenarioStudy:
    start_date: str = '2026-01-01'
    study_years: int = 20
    clock_years: int = 1
    top_n_scenarios: int = 4

    facility_sqft = 250e3
    annual_eui = 6.1  # kWh/sqft/yr
    charger_power = 100 # kW
    vehicle_hourly_energy_demand = 12  # kWh

    # Design specs
    # battery_unit_size = 210  # kWh
    unit_solar_timeseries_kw: pd.DataFrame | None = attrs.field(default=None)
    der_subpanel_load_kw: pd.DataFrame | None = attrs.field(default=None)  # Must have column 'der_subpanel_load'
    main_panel_load_kw: pd.DataFrame | None = attrs.field(default=None) # Must have column 'main_panel_load'
    available_circuit_capacity_amps: int = 100
    site_max_capacity_amps: int = 400
    site_allows_export: bool = False
    panel_voltage: int = 480
    battery_unit_size: int = 210 # kWh  # Sizing at end of life
    battery_unit_power: int = 60  # kW
    min_battery_units: int = 0
    max_battery_units: int = 5
    min_solar_units: int = 0
    max_solar_units: int = 30

    # Financial assumptions
    solar_capital_cost_per_unit: float = 156e3
    battery_capital_cost_per_unit: float = 89444 + 5000 + 11441 + 38000  # includes inverter
    solar_lifetime: int = 20
    solar_residual_value_end_of_life: float = 0.0
    battery_lifetime: int = 20
    battery_residual_value_end_of_life: float = 0.0
    discount_rate: float = 0.00
    itc_rate: float = 0.30

    # Tariff assumptions
    annual_rate_escalator = 0.06

    ## NON-init
    general_assumptions: GeneralAssumptions | None = attrs.field(default=None, init=False)
    design_spec: DesignSpec | None = attrs.field(default=None, init=False)
    tariff_spec: TariffSpec | None = attrs.field(default=None, init=False)
    financial_spec: FinancialSpec | None = attrs.field(default=None, init=False)
    result_summarizer: BasicResultSummarizer | None = attrs.field(default=None, init=False)
    ev_load: pd.Series | None = attrs.field(default=None, init=False)

    @staticmethod
    def validate_input_df(df: pd.DataFrame, target_column_name: str, target_column_synonyms: list[str]) -> pd.DataFrame:
        overlapping_columns = [c for c in target_column_synonyms if c in df.columns]
        assert len(overlapping_columns) > 0, f"Solar data must have exactly one of these columns: {target_column_synonyms}"

        try:
            df.index = pd.to_datetime(df.index)
            time_deltas = df.index.to_series().diff()
            assert len(time_deltas.unique()) == 1, "Solar data index must have consistent frequency"
        except:
            assert df.shape[0] == 8760, "Inferring time, assuming hourly starting Jan 1: Solar data must have 8760 rows"
            target_yr = 2005
            df['updated_timestamp'] = pd.date_range(start=f"{target_yr}-01-01 00:00", freq="1h",
                                                          periods=df.shape[0], tz=TIMEZONE)
            df = df.set_index('updated_timestamp')[overlapping_columns].copy()
            df = df.resample('1h').ffill().fillna(0)

        if df.max() > 1000:
            print("Assuming solar data is in W, converting to kW")
            df = df / 1000

        df = df.rename({c: target_column_name for c in overlapping_columns}, axis=1)
        return df

    def __attrs_post_init__(self):
        self.unit_solar_timeseries_kw = self.validate_input_df(self.unit_solar_timeseries_kw,
                                                      target_column_synonyms=['solar', 'ac_power_kw'],
                                                      target_column_name = 'solar',
                                                      )
        self.der_subpanel_load_kw = self.validate_input_df(self.der_subpanel_load_kw,
                                                        target_column_synonyms=['der_subpanel_load', 'load_kw', 'load'],
                                                        target_column_name='der_subpanel_load',
                                                         )
        self.main_panel_load_kw = self.validate_input_df(self.main_panel_load_kw,
                                                        target_column_synonyms=['main_panel_load', 'main_load_kw', 'load'],
                                                        target_column_name='main_panel_load',
                                                           )

        circuit_available_power_kw = self.available_circuit_capacity_amps * self.panel_voltage * np.sqrt(3) * 0.8 / 1e3  # kW
        site_max_power_kw = self.site_max_capacity_amps * self.panel_voltage * np.sqrt(3) * 0.8 / 1e3  # kW

        self.general_assumptions = GeneralAssumptions(
            start_date=self.start_date,
            study_years=self.study_years,
            optimization_type='demand_charge_tou_optimization',
            study_resolution='1h',
            optimization_clock=f'{self.clock_years}YS',
            optimization_clock_horizon=pd.DateOffset(months=self.clock_years * 12),
            # optimization_clock=None,
            # optimization_clock_horizon=None,
        )

        self.design_spec = DesignSpec(
            min_battery_units=self.min_battery_units,
            max_battery_units=self.max_battery_units,
            min_solar_units=self.min_solar_units,
            max_solar_units=self.max_solar_units,
            # Battery spec
            battery_unit_size=self.battery_unit_size,  # kWh
            battery_unit_power=self.battery_unit_power,  # kW
            battery_rt_eff=0.9025,
            backup_reserve=0.0,
            # Circuit limits
            der_subpanel_import_kw_limit=circuit_available_power_kw,  # kW
            der_subpanel_export_kw_limit=-circuit_available_power_kw,  # kW
            site_import_limit=site_max_power_kw,  # kW
            site_export_limit=-site_max_power_kw if self.site_allows_export else 0.0,  # kW
            # Solar spec
            solar_data_source="upload",
            solar_data=self.unit_solar_timeseries_kw,
            solar_first_year_degradation=0.015,
            solar_subsequent_year_degradation=0.005,
            # Circuit load data
            circuit_load_data_source="upload",
            circuit_load_data=self.der_subpanel_load_kw,
            # Non-circuit load data
            non_circuit_load_data_source="upload",  # "upload" or "flat"
            non_circuit_load_data=self.main_panel_load_kw,
        )

        self.tariff_spec = TariffSpec(rate_code="PGE_B_19_R",
                                 annual_rate_escalator=self.annual_rate_escalator,
                                 )
        self.financial_spec = FinancialSpec(study_years=self.general_assumptions.study_years,
                                            solar_capital_cost_per_unit=self.solar_capital_cost_per_unit,
                                            battery_capital_cost_per_unit=self.battery_capital_cost_per_unit,
                                            solar_lifetime=self.solar_lifetime,
                                            solar_residual_value=self.solar_residual_value_end_of_life,
                                            battery_residual_value=self.battery_residual_value_end_of_life,
                                            battery_lifetime=self.battery_lifetime,
                                            discount_rate=self.discount_rate,
                                            itc_rate=self.itc_rate,
                                            )
        self.result_summarizer = BasicResultSummarizer()

    def get_null_solar_data(self):
        base_solar_data = self.design_spec.solar_data
        null_solar_data = pd.DataFrame(0, index=base_solar_data.index, columns=base_solar_data.columns)
        return null_solar_data

    def build_baseline_scenario(self):
        design_spec_no_solar = attrs.evolve(self.design_spec,
                                            solar_data=self.get_null_solar_data(),
                                            )
        scenario_spec = ScenarioSpec(
            general_assumptions=self.general_assumptions,
            design_spec=design_spec_no_solar,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )
        runner_inputs = scenario_spec.build_optimization_runner_inputs()
        site_data = runner_inputs.design_inputs.site_data
        output_template = WrappedOptimizerOutputs()
        mock_output_df = pd.DataFrame(0, index=site_data.index, columns=output_template.result_columns)
        mock_output_df['P_grid'] = site_data[['main_panel_load', 'der_subpanel_load']].sum(axis=1)

        mock_outputs = WrappedOptimizerOutputs(
            results_df=mock_output_df,
            status='feasible',
            sizing_results = {'n_batt_blocks': 0, 'n_solar': 0},
            design_inputs=runner_inputs.design_inputs,
            financial_inputs=runner_inputs.financial_model_inputs,
        )
        results = self.result_summarizer.summarize(optimizer_results=mock_outputs)
        return results

    def run_single_scenario(self):
        assert self.min_battery_units == self.max_battery_units, "For single scenario, min and max battery units must be equal"
        assert self.min_solar_units == self.max_solar_units, "For single scenario, min and max solar units must be equal"
        scenario_spec = ScenarioSpec(
            general_assumptions=self.general_assumptions,
            design_spec=self.design_spec,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )
        runner = SizingSweepScenarioRunner(
            scenario_spec=scenario_spec,
            parallelize=True,
            n_jobs=6,
        )
        start_sim_at = time.time()
        results = runner.run_scenarios()
        end_sim_at = time.time()
        return results

    def sizing_sweep(self):
        scenario_spec = ScenarioSpec(
            general_assumptions=self.general_assumptions,
            design_spec=self.design_spec,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )
        runner = SizingSweepScenarioRunner(
            scenario_spec=scenario_spec,
            parallelize=True,
            n_jobs=6,
        )
        start_sim_at = time.time()
        results = runner.run_scenarios()
        print(f"Done with study in {time.time() - start_sim_at:.1f} seconds.")
        return results

    def sizing_optimizer(self):
        assert self.min_battery_units < self.max_battery_units or self.min_solar_units < self.max_solar_units, "For sizing sweep, must have an unconstrained variable"

        scenario_spec = ScenarioSpec(
            general_assumptions=self.general_assumptions,
            design_spec=self.design_spec,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )
        runner = TopNScenarioRunner(
            scenario_spec=scenario_spec,
            parallelize=True,
            n_jobs=6,
            n_closest=self.top_n_scenarios,
        )
        start_sim_at = time.time()
        results = runner.run_scenarios()
        print(f"Done with study in {time.time() - start_sim_at:.1f} seconds.")
        return results
