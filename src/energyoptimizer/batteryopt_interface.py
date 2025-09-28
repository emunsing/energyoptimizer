import pandas as pd
import enum

class TariffModel:
    """
    Tariffs have two components:
    - Time-of-use energy import rate: Price per kWh imported from the grid (cost)
    - Time-of-use energy export rate: price per kWh exported to the grid (revenue)
    - Demand charges: priced for the peak power which occurs during the billing cycle in the time-of-use tier. (cost)

    Demand charges:
    - General approach: We need to compose the demand charge by computing the maximum power in a billing-cycle-defined period-tier (i.e. march-peak, march-partial_peak, march-anytime; april-peak, april-partial_peak,...).  This means we need both a mapping/Series/dict of demand charge rates for each billing period-tier, and a categorical dataframe that indicates which time periods are in which billing period-tier.
    - Datetiminedex Categorical dataframe: Bool for each timeperiod if that timeindex is in the period-tier
    - demand charge price map: $/kW for peak power during the period-tier
    """

    def compute_demand_charge(self, power_series: pd.Series) -> float:
        """Compute the demand charge for a given power series."""
        # SIGN CONVENTION: Positive meter power is import, negative is export
        pass


class DesignSpec:
    """
    All the raw settings which are used to create the design inputs: i.e. this may include toggles, cases, and assumptions which are multiplied through in build_design_inputs
    This will map 1-to-1 to the user interface.

    Tabbed sections generally should be collapsing sections in the UI.

    General:
    - Study years
    - Optimization type (dropdown):
        - Self-consumption
        - TOU optimization, no demand charge optimization
        - TOU + demand charge optimization
    - Endogenous sizing: bool

    Tariff:
    - Study time index frequency
    - Rate code
    - demand charge period frequency
    - Annual rate escalator
    - include_demand_charge : bool

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
            - first-year degradation
            - subsequent-year degradation
        - Case: Use location and design specs to generate solar profile
            - location lat
            - location lon
            - azimuth
            - tilt
            - unit size (kW)
            - first-year degradation
            - subsequent-year degradation
    - Circuit load timeseries
        - Case: Upload one-year load profile
        - Case: Orange EV load modeler
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

    Cost:
    - Solar capital cost per unit
    - Solar fixed installation cost
    - Solar installation cost per unit
    - Battery capital cost per unit
    - Battery fixed installation cost
    - Battery installation cost per unit
    - Battery lifetime (years)
    - Solar lifetime (years)
    - Battery residual value at end of usable life as portion of initial battery capital cost
    - Solar residual value at end of usable life as portion of initial solar capital cost
    - Annual Inflation (deflation) in battery costs
    - Closing costs as portion of capital+install cost: float: 0.11
    - Interconnection fixed allowance: float = 10e3
    - Portion of capital cost covered by ITC: float =  0.3
    - ITC applicable to capital replacement: bool = False
    - Reference upgrade cost without DER: float = 100e3
    - Discount rate for NPV calculations: float = 0.07

    """

    def build_tariff(self) -> pd.DataFrame:

    def build_design_inputs(self) -> DesignInputs:  # TODO: forward reference
        """Build the design inputs from the design spec."""
        pass

    def build_financial_model_inputs(self) -> FinancialModelInputs:
        pass

class DesignInputs:
    """
    These ar the optimizer's inputs which are created from the DesignSpec
    Pydantic + Pandera class definitions

    Tariff_Design:
    - Tariff timeseries: site import rate, site export rate
    - Demand Charges: None or DemandChargeModel

    Design constraints
    - Circuit import limit
    - circuit export limit
    - Site import limit
    - Site export limit
    - Battery unit size
    - Battery unit power
    - Battery rt_eff
    - min number of battery units
    - max number of battery units
    - min number of solar units
    - max number of solar units

    timeseries dataframe with datetime index and columns:
    - per-unit solar production
    - circuit load timeseries
    - Non-circuit load timeseries
    """
    pass


class FinancialModelInputs:
    """
    Financials
    - Study years
    - Solar annual cost timeseries:
         - Recovered value
         - Installed cost
    - Battery annual costs:
       - Recovered value
       - Installed cost
    - ITC
    - Design cost
    - Reference upgrade cost without DER
    """

class OptimizationInputs:
    optimization_type # Enum choice
    optimization_freq # enum choice
    design_inputs: DesignInputs
    financial_model_inputs: FinancialModelInputs

class OptimizationResults:
    """
    Dataframe with the following columns:
    - solar_uncurtailed_kw
    - solar_curtailed_kw
    - battery_dispatch_kw
    - battery_energy_kwh
    - circuit_gross_load_kw
    - circuit_net_load_kw
    - non_circuit_gross_load_kw
    - meter_power_kw
    """



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