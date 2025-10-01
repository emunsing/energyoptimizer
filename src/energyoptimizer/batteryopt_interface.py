import attrs
import pandas as pd
import enum
from typing import Optional, Union
from .tariff.tariff_utils import TariffModel
from dateutil.relativedelta import relativedelta
from .batteryopt_utils import shift_copy_dataset_to_new_index

@attrs.define
class DesignInputs:
    """
    These are the optimizer's inputs which are created from the DesignSpec.
    Contains all the data and constraints needed for optimization.
    """
    # Site data with datetime index
    site_data: pd.DataFrame  # Columns: solar, circuit_load, non_circuit_load
    
    # Tariff model
    tariff_model: TariffModel
    
    # Battery constraints
    batt_rt_eff: float = 0.85
    batt_block_e_max: float = 13.5  # Battery unit size (kWh)
    batt_block_p_max: float = 5.0  # Battery unit power (kW)
    backup_reserve: float = 0.2
    
    # Circuit limits
    circuit_import_kw_limit: float = 100.0
    circuit_export_kw_limit: float = -100.0
    site_import_kw_limit: float = 100.0
    site_export_kw_limit: float = -100.0
    
    # Sizing constraints
    min_battery_units: int = 0
    max_battery_units: int = 10
    min_solar_units: int = 0
    max_solar_units: int = 10


@attrs.define
class FinancialModelInputs:
    """
    Financial model inputs for optimization.
    Contains all financial parameters needed for cost calculations.
    """
    product_cash_flows: dict[str, 'ProductCashFlows']
    discount_rate: float = 0.07
    solar_levelized_unit_cost: float = 0.0  # $/kWh
    battery_levelized_unit_cost: float = 0.0  # $/kWh
    reference_upgrade_cost: float = 100000.0  # $ without DER


@attrs.define
class DesignSpec:
    """
    All the raw settings which are used to create the design inputs: i.e. this may include toggles, cases, and assumptions which are multiplied through in build_design_inputs
    This will map 1-to-1 to the user interface.

    Tabbed sections generally should be collapsing sections in the UI.

    Design inputs:
    - Number of battery units: list[int, int] min, max
    - Number of solar units: list[int, int] min, max
    - Battery unit size: float
    - Battery unit power: float
    - Battery rt_eff: float
    - Circuit import limit
    - circuit export limit
    - Site import limit
    - Site export limit
    - solar unit timeseries:
        - Case: Upload full-study solar profile
            - CSV with datetime index and solar column
        - Case: Upload year-1 solar profile
            - CSV with datetime index and solar column
            - first-year degradation: 0.015
            - subsequent-year degradation: 0.008
        - Case: Use location and design specs to generate solar profile  # NOT IMPLEMENTED
            - location lat
            - location lon
            - azimuth
            - tilt
            - unit size (kW)
            - first-year degradation
            - subsequent-year degradation
    - Circuit load timeseries
        - Case: Upload one-year load profile
        - Case: Orange EV load modeler  # NOT IMPLEMENTED
            - Charger peak power output
            - EVs per charger
            - EV peak charge power
            - EV hourly energy demand
            - EV battery capacity (kWh)
            - EV_charge_at_soe_percent  # Start seeking charge at this state of energy, but can continue operating if the charger is not available
            - EV_idle_at_soe_percent  # Stop operating and go idle if the state of energy drops to this level
            - earliest_shift_stat_hr
            - latest_shift_end_hr
            - scheduled_charge_breaks: list of (start_time, end_time)
    - Non-circuit load timeseries:
        - Case: Upload one-year load profile
        - Case: Flat load profile
          - Facility square footage
          - Annual energy use intensity (EUI) in kWh/sqft-yr
    """
    # Sizing constraints
    min_battery_units: int = 0
    max_battery_units: int = 10
    min_solar_units: int = 0
    max_solar_units: int = 10
    
    # Battery specifications
    battery_unit_size: float = 13.5  # kWh
    battery_unit_power: float = 5.0  # kW
    battery_rt_eff: float = 0.85
    backup_reserve: float = 0.2
    
    # Circuit limits
    circuit_import_limit: float = 100.0  # kW
    circuit_export_limit: float = -100.0  # kW
    site_import_limit: float = 100.0  # kW
    site_export_limit: float = -100.0  # kW
    
    # Solar data source configuration
    solar_data_source: str = "upload"  # "upload", "year1", "generated"
    solar_data: Optional[pd.DataFrame] = None  # For uploaded data
    solar_first_year_degradation: float = 0.015
    solar_subsequent_year_degradation: float = 0.008
    
    # Circuit load data source configuration
    circuit_load_data_source: str = "upload"  # "upload", "ev_modeler"
    circuit_load_data: Optional[pd.DataFrame] = None  # For uploaded data
    
    # Non-circuit load data source configuration
    non_circuit_load_data_source: str = "upload"  # "upload", "flat"
    non_circuit_load_data: Optional[pd.DataFrame] = None  # For uploaded data
    facility_sqft: Optional[float] = None  # For flat load profile
    annual_eui: Optional[float] = None  # kWh/sqft-yr for flat load profile
    
    def build_solar_timeseries(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Build the solar timeseries from the design spec."""
        if self.solar_data_source == "upload" and self.solar_data is not None:
            # Use uploaded data directly - extract the series from DataFrame
            solar_series = self.solar_data.loc['solar']
            if solar_series.index[0] <= time_index[0] and solar_series.index[-1] >= time_index[-1]:
                # Supplied series is fully covers the study
                solar_series = solar_series.loc[time_index[0]:time_index[-1]]
                result = solar_series.reindex(time_index, method='nearest')
            else:
                solar_series = shift_copy_dataset_to_new_index(input_dataset=solar_series,
                                                               new_time_index=time_index,)

                # Model solar panel decay for years not covered by the supplied dataset
                input_series_duration_years = relativedelta(solar_series.index[-1] + solar_series.index.freq, solar_series.index[0]).years
                experiment_duration_years = relativedelta(time_index[-1] + time_index.freq, time_index[0]).years

                # Linear solar panel decay after the initial decay
                solar_decay_multipliers = [1.0] + [1.0 - self.solar_first_year_degradation - self.solar_subsequent_year_degradation * i for i in
                 range(experiment_duration_years - 1)]

                for yr in range(input_series_duration_years, experiment_duration_years):
                    yr_start = solar_series.index[0] + relativedelta(years=yr)
                    yr_end = solar_series.index[0] + relativedelta(years=yr+1) - pd.DateOffset(minutes=1)
                    multiplier = solar_decay_multipliers[yr]
                    solar_series.loc[yr_start:yr_end] *= multiplier
                result = solar_series
            return result
        
        else:
            # Default: return zeros if no data provided
            raise ValueError("No solar data provided")
    
    def build_circuit_load_timeseries(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Build the circuit load timeseries from the design spec."""
        if self.circuit_load_data_source == "upload" and self.circuit_load_data is not None:
            # Use uploaded data directly - extract the series from DataFrame
            load_series = self.circuit_load_data['load']

            reindexed_load_series = shift_copy_dataset_to_new_index(input_dataset=load_series,
                                                                    new_time_index=time_index,)
            return reindexed_load_series
        
        else:
            # Default: return zeros if no data provided
            raise ValueError("No circuit load data provided")
    
    def build_non_circuit_load_timeseries(self, time_index: pd.DatetimeIndex) -> pd.Series:
        """Build the non-circuit load timeseries from the design spec."""
        if self.non_circuit_load_data_source == "upload" and self.non_circuit_load_data is not None:
            # Use uploaded data directly - extract the series from DataFrame
            load_series = self.non_circuit_load_data.loc['site_load']  # Get first column as Series
            # TODO: This needs to be fixed; we need to copy/shift the year-1 data to subsequent years
            result = shift_copy_dataset_to_new_index(input_dataset=load_series,
                                                     new_time_index=time_index,
                                                     )
            result.name = 'non_circuit_load'
            return result
        
        elif self.non_circuit_load_data_source == "flat" and self.facility_sqft is not None and self.annual_eui is not None:
            # Generate flat load profile
            annual_energy = self.facility_sqft * self.annual_eui
            hourly_energy = annual_energy / 8760  # kWh/hr
            return pd.Series(hourly_energy, index=time_index, name='non_circuit_load')
        
        else:
            # Default: return zeros if no data provided
            return pd.Series(0.0, index=time_index, name='non_circuit_load')


@attrs.define
class ProductFinancialSpec:
    fixed_upfront_installation_cost: float
    capital_cost_per_unit: float
    installationrepair_labor_cost_per_unit: float
    lifetime_years: int
    residual_value: float  # portion of initial cost
    annual_inflation_rate: float  # Deflation if negative
    itc_rate: float  # Investment Tax Credit rate
    apply_inflation_to_labor: bool = False
    itc_applies_to_replacement: bool = False


@attrs.define
class ProductCashFlows:
    fixed_upfront_installation_cost: float
    unit_cash_flows: pd.DataFrame
    unit_annualized_cost: float


@attrs.define
class FinancialSpec:
    """
    Cost:
    - Solar fixed upfront installation cost (doesn't scale with number of units): 0
    - Solar capital/replacement cost per unit
    - Solar labor cost per unit for installation or replacement:
    - Solar lifetime (years)
    - Solar residual value at end of usable life as portion of initial solar capital cost
    - Solar expected replacement cost inflation/deprectiation rate (annual): 0
    - Portion of solar capital cost covered by ITC: float =  0.3
    - ITC applicable to capital replacement: bool = False


    - Battery fixed upfront installation cost (doesn't scale with number of units): 0
    - Battery capital/replacement cost per unit
    - Battery installation/replacement labor cost per unit
    - Battery lifetime (years)
    - Battery residual value at end of usable life as portion of initial battery capital cost
    - Battery expected replacement cost inflation/deprectiation rate (annual): -0.05
    - Portion of battery capital cost covered by ITC: float =  0.3
    - ITC applicable to capital replacement: bool = False

    - Loan annual interest rate: float = 0.085
    - Closing costs as portion of capital+install cost: float: 0.11
    - Interconnection fixed allowance: float = 10e3

    # Inputs which are used only for results:
    - Reference upgrade cost without DER: float = 100e3
    - Discount rate for NPV calculations for results only: float = 0.07

    ## Computing the FinancialInputs from the FinancialSpec:
    To create the FinancialInputs, we need to compute the levelized annual cost of purchasing a unit of solar or battery
    For each technology, we need to compute:
    - Number of replacements over the study period
    - Present value of the up-front costs
    - For each time we need to install/replace the technology:
        - Present value of the replacement costs, adjusted for inflation and ITC
        - Present value of the residual value at replacement or at the study end, if the study ends before the end of the lifetime

    - With those present values, we need to add the closing costs (as an inflator on the full NPV)
    - Based on that inflated NPV, compute the financing cost (using annuity formula) for the technology, using the loan annual interest rate
    - Levelized annual cost (12 * monthly financing cost)
    """
    # Study parameters
    study_years: int = 10
    discount_rate: float = 0.07
    
    # Solar costs
    solar_capital_cost_per_unit: float = 3000.0  # $/kW
    solar_installationrepair_labor_cost_per_unit: float = 0.0  # $/kW
    solar_fixed_upfront_installation_cost: float = 0.0  # $
    solar_lifetime: int = 25  # years
    solar_residual_value: float = 0.0  # portion of initial cost
    solar_annual_inflation_rate: float = 0.0
    
    # Battery costs
    battery_capital_cost_per_unit: float = 1000.0  # $/kWh
    battery_installation_cost_per_unit: float = 0.0  # $/kWh
    battery_fixed_upfront_installation_cost: float = 0.0  # $
    battery_lifetime: int = 10  # years
    battery_residual_value: float = 0.0  # portion of initial cost
    battery_annual_inflation_rate: float = 0.0  # Deflation if negative
    
    # Financial incentives
    itc_rate: float = 0.3  # Investment Tax Credit rate
    itc_applies_to_replacement: bool = False
    apply_inflation_to_labor: bool = False
    
    # Other costs
    closing_costs_rate: float = 0.11  # portion of capital+install cost
    # These costs are not per-unit, and so do not fit well into the
    interconnection_allowance: float = 10000.0  # $
    design_cost: float = 0.0  # $
    reference_upgrade_cost: float = 100000.0  # $ without DER

    ## Not accessible with init ##
    solar_unit_cash_flows: ProductCashFlows | None = attrs.field(init=False, default=None)
    battery_unit_cash_flows: ProductCashFlows | None = attrs.field(init=False, default=None)
    total_cash_flows: pd.DataFrame | None = attrs.field(init=False, default=None)

    @staticmethod
    def _get_zero_cash_flow_df(study_years):
        cash_flow_df_cols = ['capital_cost', 'residual_value', 'soft_cost', 'itc_credit',
                             'energy_charges', 'demand_charges', 'total_charges', 'ppa_cost']
        return pd.DataFrame(0.0, index=range(study_years), columns=cash_flow_df_cols)

    def _initialize_cash_flow_df(self, study_years):
        if not hasattr(self, 'cash_flow_df') or self.cash_flow_df is None or self.cash_flow_df.empty:
            self.cash_flow_df = self._get_zero_cash_flow_df(study_years)

    @staticmethod
    def annuity_payment(r, n):
        return (r * (1.0 + r) ** n) / (((1.0 + r) ** n) - 1.0)

    @staticmethod
    def get_pv_of_cash_flows(cash_flow_df: pd.DataFrame, discount_rate: float) -> float:
        cash_flow_series = cash_flow_df.sum(axis=1)
        # Discount cash flows to present value
        discount_factors = [(1.0 + discount_rate) ** i for i in range(cash_flow_df.shape[0])]
        discounted_cash_flows = cash_flow_series / discount_factors
        return discounted_cash_flows.sum()

    def compute_unit_replacement_cash_flows(self,
                                        study_years: int,
                                        product_spec: ProductFinancialSpec
    ) -> pd.DataFrame:

        cash_flows = self._get_zero_cash_flow_df(study_years)

        # Replacement times in years (including t=0 for initial install)
        assert product_spec.lifetime_years > 0, "Lifetime years must be greater than 0"
        replacement_times: list[int] = []
        t = 0
        while t < study_years:
            replacement_times.append(t)
            t += product_spec.lifetime_years

        for t in replacement_times:
            # Inflation applies to replacement capital cost over t years (not to initial t=0)
            inflation_factor = (1.0 + product_spec.annual_inflation_rate) ** t
            future_capital_cost = product_spec.capital_cost_per_unit * inflation_factor
            if product_spec.apply_inflation_to_labor:
                future_labor_cost = product_spec.installationrepair_labor_cost_per_unit * inflation_factor
            else:
                future_labor_cost = product_spec.installationrepair_labor_cost_per_unit
            base_cost = future_capital_cost + future_labor_cost
            apply_itc = (t == 0) or product_spec.itc_applies_to_replacement
            if apply_itc:
                base_cost *= (1- product_spec.itc_rate)

            cash_flows.at[t, 'capital_cost'] -= future_capital_cost

            # Compute the future residual value for this unit at either the end of the current lifetime, or the end of the study
            retirement_year = t + product_spec.lifetime_years
            if retirement_year > study_years:
                retirement_year = study_years

            portion_depreciation_period = (retirement_year - t) / product_spec.lifetime_years
            depreciation_cost = base_cost * (1 - product_spec.residual_value) * portion_depreciation_period
            residual_value = base_cost - depreciation_cost
            cash_flows.at[retirement_year, 'residual_value'] += residual_value  # Credit at retirement year


        return cash_flows

    def compute_product_cash_flows(self,
                                       study_years: int,
                                       interest_rate: float,
                                       closing_costs_rate: float,
                                       product_spec: ProductFinancialSpec
                                       ):

        product_cash_flows = self.compute_unit_replacement_cash_flows(
            study_years=study_years,
            product_spec=product_spec
        )

        total_pv = self.get_pv_of_cash_flows(product_cash_flows, discount_rate=interest_rate)
        closing_costs = total_pv * closing_costs_rate
        product_cash_flows.at[0, 'soft_cost'] += closing_costs

        effective_monthly_interest_rate = (1.0 + interest_rate) ** (1.0 / 12.0) - 1.0
        annuity_multiplier = self.annuity_payment(r=effective_monthly_interest_rate,
                                            n=study_years * 12)
        total_unit_pv = total_pv + closing_costs
        monthly_payment = total_unit_pv * annuity_multiplier
        annual_payment = monthly_payment * 12.0

        return ProductCashFlows(
            fixed_upfront_installation_cost=product_spec.fixed_upfront_installation_cost,
            unit_cash_flows=product_cash_flows,
            unit_annualized_cost=annual_payment
        )

    def build_financial_model_inputs(self, study_years: int) -> FinancialModelInputs:
        self._initialize_cash_flow_df(study_years)

        solar_spec = ProductFinancialSpec(
            fixed_upfront_installation_cost=self.solar_fixed_upfront_installation_cost,
            capital_cost_per_unit=self.solar_capital_cost_per_unit,
            installationrepair_labor_cost_per_unit=self.solar_installationrepair_labor_cost_per_unit,
            lifetime_years=self.solar_lifetime,
            residual_value=self.solar_residual_value,
            annual_inflation_rate=self.solar_annual_inflation_rate,
            itc_rate=self.itc_rate,
            apply_inflation_to_labor=self.apply_inflation_to_labor,
            itc_applies_to_replacement=self.itc_applies_to_replacement
        )

        battery_spec = ProductFinancialSpec(
            fixed_upfront_installation_cost=self.battery_fixed_upfront_installation_cost,
            capital_cost_per_unit=self.battery_capital_cost_per_unit,
            installationrepair_labor_cost_per_unit=self.battery_installation_cost_per_unit,
            lifetime_years=self.battery_lifetime,
            residual_value=self.battery_residual_value,
            annual_inflation_rate=self.battery_annual_inflation_rate,
            itc_rate=self.itc_rate,
            apply_inflation_to_labor=self.apply_inflation_to_labor,
            itc_applies_to_replacement=self.itc_applies_to_replacement
        )

        self.solar_unit_cash_flows = self.compute_product_cash_flows(
            study_years=study_years,
            interest_rate=self.discount_rate,
            closing_costs_rate=self.closing_costs_rate,
            product_spec=solar_spec
        )

        self.battery_unit_cash_flows = self.compute_product_cash_flows(
            study_years=study_years,
            interest_rate=self.discount_rate,
            closing_costs_rate=self.closing_costs_rate,
            product_spec=battery_spec
        )

        return FinancialModelInputs(
            product_cash_flows = {'solar': self.solar_unit_cash_flows, 'battery': self.battery_unit_cash_flows},
            discount_rate=self.discount_rate,
            solar_levelized_unit_cost=self.solar_unit_cash_flows.unit_annualized_cost,
            battery_levelized_unit_cost=self.battery_unit_cash_flows.unit_annualized_cost,
            reference_upgrade_cost=self.reference_upgrade_cost
        )


@attrs.define
class GeneralAssumptions:
    """
    General:
    - Study years
    - Study time index frequency
    - Optimization type (dropdown):
        - Self-consumption
        - TOU optimization, no demand charge optimization
        - TOU + demand charge optimization
    - Endogenous sizing: bool
    """
    start_date: Optional[Union[str, pd.Timestamp]] = None
    timezone: str = 'US/Pacific'
    study_years: int = 10
    endogenous_sizing: bool = False
    optimization_type: str = 'self_consumption'  # 'self_consumption', 'tou_optimization', 'tou_endogenous_sizing'
    study_resolution: str = '1H'  # e.g., '1H', '15T'
    end_date: Optional[pd.Timestamp] = None

    def __attrs_post_init__(self):
        if self.start_date is None:
            self.start_date = pd.Timestamp.now(tz=self.timezone).normalize()
        elif isinstance(self.start_date, str):
            self.start_date = pd.Timestamp(self.start_date, tz=self.timezone)
        
        if self.end_date is None:
            self.end_date = self.start_date + pd.DateOffset(years=self.study_years)


@attrs.define
class TariffSpec:
    """
    Tariff specifications for the optimization.
    """
    rate_code: str
    annual_rate_escalator: float = 0.0
    include_demand_charge: bool = True
    demand_charge_billing_frequency: str = 'month_last_day_of_month'

    def build_tariff(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> TariffModel:
        """Build the tariff from the tariff spec."""
        return TariffModel('tariffs.yaml', self.rate_code, start_date, end_date, self.annual_rate_escalator)



@attrs.define
class OptimizationClock:
    """We may want to create non-overlapping optimization windows (typically for design),
    or we may want rolling optimization with a long forecasting window, but frequent re-optimization (model predictive control archetype).
    """
    frequency: str | pd.DateOffset  # e.g., '1D' for daily, '1H' for hourly
    horizon: Optional[pd.DateOffset] = None  # e.g., '7D' for 7 days, '1D' for 1 day
    lookback: Optional[pd.DateOffset] = None

    def get_intervals(self, start: pd.Timestamp, end: pd.Timestamp) -> list[
        tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate a list of (optimize_at, data_from, data_until) tuples for optimization intervals.

        Args:
            start: Start timestamp for the optimization period
            end: End timestamp for the optimization period

        Returns:
            List of tuples containing (optimize_at, data_from, data_until) for each optimization interval
        """
        # Generate optimization timestamps using pd.date_range
        optimize_times = pd.date_range(start=start, end=end, freq=self.frequency)

        intervals = []
        for optimize_at in optimize_times:
            # Use smart defaults: if lookback/horizon is None, use the full period
            data_from = max(start, optimize_at - (self.lookback or pd.DateOffset(0)))
            data_until = min(end, optimize_at + (self.horizon or pd.DateOffset(0)))

            intervals.append((optimize_at, data_from, data_until))

        return intervals



class OptimizationType(enum.Enum):
    """Enum for different optimization types available in the system."""
    SELF_CONSUMPTION = "self_consumption"
    TOU_OPTIMIZATION = "tou_optimization"
    TOU_ENDOGENOUS_SIZING = "tou_endogenous_sizing"


@attrs.define
class OptimizationRunnerInputs:
    optimization_type: OptimizationType
    optimization_start: pd.Timestamp
    optimization_end: pd.Timestamp
    design_inputs: 'DesignInputs'
    financial_model_inputs: 'FinancialModelInputs'
    optimization_clock: Optional[OptimizationClock] = None
    parallelize: bool = True

@attrs.define
class ScenarioSpec:
    """
    Complete scenario specification that combines all aspects of the optimization.
    """
    general_assumptions: GeneralAssumptions
    design_spec: DesignSpec
    tariff_spec: TariffSpec
    financial_spec: FinancialSpec
    study_years: int | None = None

    def build_tariff(self) -> TariffModel:
        """Build the tariff from the tariff spec."""
        return self.tariff_spec.build_tariff(
            self.general_assumptions.start_date,
            self.general_assumptions.end_date
        )

    def build_design_inputs(self) -> DesignInputs:
        """Build the design inputs from the design spec."""
        # Create time index - ensure timezone consistency
        start_date = self.general_assumptions.start_date
        if hasattr(start_date, 'tz_localize'):
            # If start_date is timezone-naive, localize it
            if start_date.tz is None:
                start_date = start_date.tz_localize(self.general_assumptions.timezone)
        elif isinstance(start_date, str):
            start_date = pd.Timestamp(start_date, tz=self.general_assumptions.timezone)
        
        time_index = pd.date_range(
            start=start_date,
            end=self.general_assumptions.end_date,
            freq=self.general_assumptions.study_resolution
        )
        
        # Build timeseries data
        solar_data = self.design_spec.build_solar_timeseries(time_index)
        circuit_load_data = self.design_spec.build_circuit_load_timeseries(time_index)
        non_circuit_load_data = self.design_spec.build_non_circuit_load_timeseries(time_index)
        
        # Combine into site data DataFrame
        site_data = pd.DataFrame({
            'solar': solar_data,
            'circuit_load': circuit_load_data,
            'non_circuit_load': non_circuit_load_data
        }, index=time_index)
        
        # Build tariff model
        tariff_model = self.build_tariff()
        
        return DesignInputs(
            site_data=site_data,
            tariff_model=tariff_model,
            batt_rt_eff=self.design_spec.battery_rt_eff,
            batt_block_e_max=self.design_spec.battery_unit_size,
            batt_block_p_max=self.design_spec.battery_unit_power,
            backup_reserve=self.design_spec.backup_reserve,
            circuit_import_kw_limit=self.design_spec.circuit_import_limit,
            circuit_export_kw_limit=self.design_spec.circuit_export_limit,
            site_import_kw_limit=self.design_spec.site_import_limit,
            site_export_kw_limit=self.design_spec.site_export_limit,
            min_battery_units=self.design_spec.min_battery_units,
            max_battery_units=self.design_spec.max_battery_units,
            min_solar_units=self.design_spec.min_solar_units,
            max_solar_units=self.design_spec.max_solar_units
        )

    def build_financial_model_inputs(self) -> FinancialModelInputs:
        """Build the financial model inputs from the financial spec."""
        return self.financial_spec.build_financial_model_inputs(
            study_years=self.general_assumptions.study_years,
        )

    def build_optimization_runner_inputs(self) -> OptimizationRunnerInputs:
        """Build the complete optimization runner inputs from the scenario spec."""
        design_inputs = self.build_design_inputs()
        financial_model_inputs = self.build_financial_model_inputs()

        optimization_type = OptimizationType(self.general_assumptions.optimization_type)

        optimization_clock = OptimizationClock(frequency='1Y',
                                               horizon=None,
                                               lookback=None)

        return OptimizationRunnerInputs(
            optimization_type=optimization_type,
            optimization_start=self.general_assumptions.start_date,
            optimization_end=self.general_assumptions.end_date,
            design_inputs=design_inputs,
            financial_model_inputs=financial_model_inputs,
            optimization_clock=optimization_clock,
            parallelize=False
        )



class SummaryScalarStats:
    """Pydantic + Pandera class definitions
    - total_solar_production_kwh: float
    - average_annual_imported_kwh: float
    - average_annual_exported_kwh: float
    - average_annual_load_kwh: float
    - average_annual_solar_production_kwh: float
    - average_annual_solar_curtailment_kwh: float
    """

class AnnualFinancialStats:
    """Pydantic + Pandera class definitions
    - annual_import_cost: float
    - annual_export_revenue: float

    """