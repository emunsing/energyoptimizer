import attrs
import cvxpy as cp
import pandas as pd
import numpy as np
import time
import scipy.sparse as sps

# from .batteryopt_interface import TariffModel  # Commented out due to incomplete implementation
from src.energyoptimizer.batteryopt_utils import MIN_DT

@attrs.define
class OptimizationInputs:
    """
    Input: OptimizationRunnerInputs
    Structured for optimizer: OptimizationInputs

    This imagines a configuration where solar is behind-the-meter on a subpanel, which may have a constrained
    connection to the grid
    
    grid <-> main_panel <-> der_subpanel: solar, battery, der_subpanel_load

    """
    start: pd.Timestamp
    end: pd.Timestamp
    site_data: pd.DataFrame  # Columns: solar, circuit_load, non_circuit_load
    tariff_model: 'object'  # Any object with get_tariff_data method
    batt_rt_eff: float = 0.85
    batt_block_e_max: float = 13.5
    batt_block_p_max: float = 5.0
    backup_reserve: float = 0.2
    batt_starting_soe: float | None = 0.5
    der_subpanel_import_kw_limit: float = 100.0
    der_subpanel_export_kw_limit: float = -100.0
    site_import_kw_limit: float = 100.0
    site_export_kw_limit: float = -100.0
    # Endogenous sizing parameters
    solar_annualized_cost_per_kw: float = 0.15  # 3.0 / 20
    batt_annualized_cost_per_unit: float = 1000.0
    integer_problem: bool = False


@attrs.define
class OptimizerOutputs:
    """Standardized output format for all optimizers."""
    results_df: pd.DataFrame | None = attrs.field(default=None)
    status: str | None = attrs.field(default=None)
    sizing_results: dict = attrs.field(factory=lambda: {"n_batt_blocks": 1, "n_solar": 1})
    _intermediate_sizing_results = attrs.field(init=False, factory=lambda: {"n_batt_blocks": [], "n_solar": []})
    _intermediate_status_list = attrs.field(init=False, factory=list)

    def get_results(self) -> pd.DataFrame:
        """Get the optimization results DataFrame."""
        return self.results_df

    def get_sizing_results(self) -> dict:
        """Get sizing results if available (for endogenous sizing optimizers)."""
        return self.sizing_results

    def append(self, new_results: 'OptimizerOutputs'):
        if self.results_df is None:
            # Assume this is a cold start; we should just return the new results
            self.results_df = new_results.results_df
            self.status = new_results.status
            self.sizing_results = new_results.sizing_results
        else:
            new_results_start_at = new_results.results_df.index[0]
            keep_results = self.results_df.loc[:new_results_start_at - MIN_DT, :]
            self.results_df = pd.concat([keep_results, new_results.results_df])

            self._intermediate_status_list += [new_results.status]
            self.status = new_results.status

            assert sorted(self.sizing_results.keys()) == sorted(new_results.sizing_results.keys()), "Incompatible sizing results keys"
            for k, v in new_results.sizing_results.items():
                self._intermediate_sizing_results[k] += [v]
            self.sizing_results = new_results.sizing_results

    def trim(self, start_time, end_time):
        if self.results_df is not None:
            self.results_df = self.results_df.loc[start_time:end_time, :]

    def finalize(self, start_time, end_time):
        self.trim(start_time, end_time)
        self.status = self._intermediate_status_list[-1]  # No-op for now
        for k, v in self._intermediate_sizing_results.items():
            self.sizing_results[k] = np.mean(v)


def tou_optimization(opt_inputs: OptimizationInputs) -> OptimizerOutputs:
    # Extract parameters from OptimizationInputs
    site_data = opt_inputs.site_data
    batt_rt_eff = opt_inputs.batt_rt_eff
    batt_e_max = opt_inputs.batt_block_e_max
    batt_p_max = opt_inputs.batt_block_p_max
    backup_reserve = opt_inputs.backup_reserve
    import_kw_limit = opt_inputs.der_subpanel_import_kw_limit
    export_kw_limit = opt_inputs.der_subpanel_export_kw_limit

    tariff = opt_inputs.tariff_model.tariff_timeseries.copy()
    tariff = tariff.loc[site_data.index[0]:site_data.index[-1],:]

    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
    time_intervals = site_data.index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600

    px_sell_dt = tariff['energy_export_rate_kwh'] * dt
    px_buy_dt = tariff['energy_import_rate_kwh'] * dt

    oneway_eff = np.sqrt(batt_rt_eff)
    starting_soe = opt_inputs.batt_starting_soe if opt_inputs.batt_starting_soe is not None else backup_reserve
    starting_energy_kwh = starting_soe * batt_e_max
    e_min = backup_reserve * batt_e_max

    n = site_data.shape[0]
    E_0 = starting_energy_kwh

    E_transition = sps.hstack([sps.eye(n, format="csr"), sps.csr_matrix((n, 1))], format="csr")
    # E_transition = np.hstack([np.eye(n), np.zeros(n).reshape(-1,1)])

    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
    P_subpanel_import = cp.Variable(n)
    P_subpanel_export = cp.Variable(n)
    solar_post_curtailment = cp.Variable(n, nonneg=True)
    E = cp.Variable(n+1)

    # Power flows are all AC, and are signed relative to the bus: injections to the bus are positive, withdrawals/exports from the bus are negative

    constraints = [-batt_p_max <= P_batt_charge,
                P_batt_charge <= 0,
                0 <= P_batt_discharge,
                P_batt_discharge <= batt_p_max,
                0 <= P_grid_buy,
                P_grid_buy <= import_kw_limit,
                P_grid_sell <= 0,
                export_kw_limit <= P_grid_sell,
                0 <= P_subpanel_import,
                P_subpanel_import <= opt_inputs.der_subpanel_import_kw_limit,
                P_subpanel_export <= 0,
                opt_inputs.der_subpanel_export_kw_limit <= P_subpanel_export,
                P_grid_buy + P_grid_sell - site_data['main_panel_load'] + P_subpanel_import + P_subpanel_export == 0,
                e_min <= E,
                E <= batt_e_max,
                solar_post_curtailment >= 0,
                solar_post_curtailment <= site_data['solar'],
                E[1:] == E_transition @ E - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - site_data['der_subpanel_load'] - site_data['main_panel_load'] + solar_post_curtailment == 0,
                E[0] == E_0
                ]

    # Note: Costs are positive, revenues negative, tariff is always positive but export flows are negative
    obj = cp.Minimize(P_grid_sell @ px_sell_dt + P_grid_buy @ px_buy_dt)

    prob = cp.Problem(obj, constraints)

    opt_start = time.time()
    prob.solve()
    if prob.status == cp.INFEASIBLE:
        raise AssertionError("Infeasible")
    print(f"Optimization done in {time.time() - opt_start :.3f} seconds")

    res = pd.DataFrame.from_dict({'P_batt': P_batt_charge.value + P_batt_discharge.value,
                                  'P_grid': P_grid_buy.value + P_grid_sell.value,
                                  'P_subpanel': P_subpanel_import.value + P_subpanel_export.value,
                                  'E': E[1:].value,
                                  'solar_post_curtailment': solar_post_curtailment.value
                                  }).set_index(site_data.index)
    return OptimizerOutputs(results_df=res)

def single_panel_self_consumption(opt_inputs: OptimizationInputs) -> OptimizerOutputs:
    """
    Simple self-consumption battery dispatch algorithm.
    Returns DataFrame with columns: P_batt, P_grid, E, solar_post_curtailment
    """
    # Extract parameters from OptimizationInputs
    site_data = opt_inputs.site_data
    tariff = opt_inputs.tariff_model.tariff_timeseries.copy()
    tariff = tariff.loc[site_data.index[0]:site_data.index[-1],:]
    batt_rt_eff = opt_inputs.batt_rt_eff
    batt_e_max = opt_inputs.batt_block_e_max
    batt_p_max = opt_inputs.batt_block_p_max
    backup_reserve = opt_inputs.backup_reserve
    import_kw_limit = opt_inputs.der_subpanel_import_kw_limit
    export_kw_limit = opt_inputs.der_subpanel_export_kw_limit

    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
    time_intervals = site_data.index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600

    starting_soe = opt_inputs.batt_starting_soe if opt_inputs.batt_starting_soe is not None else backup_reserve
    starting_energy_kwh = starting_soe * batt_e_max

    oneway_eff = np.sqrt(batt_rt_eff)
    e_min = backup_reserve * batt_e_max
    n = site_data.shape[0]
    E = np.zeros(n+1)
    E[0] = starting_energy_kwh
    P_batt = np.zeros(n)
    P_grid = np.zeros(n)
    solar_post_curtailment = np.zeros(n)

    for t in range(n):
        load = site_data['load'].iloc[t]
        solar = site_data['solar'].iloc[t]
        net_load = load - solar  # positive: excess solar, negative: deficit
        batt_e = E[t]
        # Excess solar: try to charge battery
        if net_load < 0:
            # Max possible charge (limited by battery power and available solar)
            max_charge_power = min(batt_p_max, -net_load, (batt_e_max - batt_e) / (dt * oneway_eff))
            charge_power = max(0, max_charge_power)
            # Update battery state
            P_batt[t] = -charge_power  # charging is negative
            E[t+1] = batt_e + charge_power * oneway_eff * dt
            # Remaining solar after battery charging
            remaining_solar = net_load + charge_power
            # Export to grid up to export_kw_limit (which is negative)
            export_power = min(0, max(remaining_solar, export_kw_limit))
            curtailed_solar = abs(remaining_solar - export_power)
            solar_post_curtailment[t] = solar - curtailed_solar
            P_grid[t] = export_power
            # Any remaining solar is curtailed
        else:
            # Deficit: try to discharge battery
            max_discharge_power = min(batt_p_max, net_load, (batt_e - e_min) * oneway_eff / dt)
            discharge_power = max(0, max_discharge_power)
            # Update battery state
            P_batt[t] = discharge_power  # discharging is positive
            E[t+1] = batt_e - discharge_power / oneway_eff * dt
            # Remaining deficit after battery discharge
            remaining_deficit = net_load - discharge_power
            # Import from grid up to import_kw_limit
            import_power = min(import_kw_limit, remaining_deficit)
            P_grid[t] = import_power
            # No curtailment in deficit case
            solar_post_curtailment[t] = solar
    res = pd.DataFrame.from_dict({'P_batt': P_batt,
                                  'P_grid': P_grid,
                                  'E': E[1:],
                                  'solar_post_curtailment': solar_post_curtailment
                                  }).set_index(site_data.index)
    return OptimizerOutputs(results_df=res)

def subpanel_self_consumption(opt_inputs: OptimizationInputs) -> OptimizerOutputs:
    """
    Simple self-consumption battery dispatch algorithm.
    Returns DataFrame with columns: P_batt, P_grid, E, solar_post_curtailment

    If we have solar,
    - Feed the subpanel load first
    - Feed the main panel load second, up to min(der_subpanel_export_kw_limit, main_panel_load)
    - If there is excess, charge the battery
    - If there is still excess, and we aren't at the subpanel export limit, export to grid within site_export_limit
        - Note: This won't happen if the der_subpanel_export_kw_limit is binding
    - Any remaining excess is curtailed
    """
    # Extract parameters from OptimizationInputs
    site_data = opt_inputs.site_data
    tariff = opt_inputs.tariff_model.tariff_timeseries.copy()
    tariff = tariff.loc[site_data.index[0]:site_data.index[-1],:]
    batt_rt_eff = opt_inputs.batt_rt_eff
    batt_e_max = opt_inputs.batt_block_e_max
    batt_p_max = opt_inputs.batt_block_p_max
    backup_reserve = opt_inputs.backup_reserve

    if len(site_data) == 1:
        # Testing infrastructure; single timestep
        dt = 1.0
    else:
        assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
        time_intervals = site_data.index.diff()[1:].unique()
        assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
        dt = time_intervals[0].total_seconds() / 3600

    starting_soe = opt_inputs.batt_starting_soe if opt_inputs.batt_starting_soe is not None else backup_reserve
    starting_energy_kwh = starting_soe * batt_e_max

    oneway_eff = np.sqrt(batt_rt_eff)
    e_min = backup_reserve * batt_e_max
    n = site_data.shape[0]
    E = np.zeros(n+1)
    E[0] = starting_energy_kwh
    P_batt = np.zeros(n)
    P_subpanel = np.zeros(n)
    P_grid = np.zeros(n)
    solar_post_curtailment = np.zeros(n)
    feasible = True

    for t in range(n):
        der_subpanel_load = site_data['der_subpanel_load'].iloc[t]
        main_panel_load = site_data['main_panel_load'].iloc[t]
        servable_main_load = min(-opt_inputs.der_subpanel_export_kw_limit, main_panel_load)
        solar = site_data['solar'].iloc[t]
        subpanel_net_load = der_subpanel_load - solar  # positive: excess solar, negative: deficit
        servable_net_load = der_subpanel_load + servable_main_load - solar
        effective_net_load = servable_net_load
        batt_e = E[t]

        if not feasible:
            # Fill remainder of timeseries with NaNs after the breaking timestamp
            P_batt[t] = np.nan
            P_subpanel[t] = np.nan
            P_grid[t] = np.nan
            solar_post_curtailment[t] = np.nan
            E[t+1] = np.nan

        elif effective_net_load < 0:
            # Excess solar: try to charge battery

            # Max possible charge (limited by battery power and available solar)
            max_charge_power = min(batt_p_max, -effective_net_load, (batt_e_max - batt_e) / (dt * oneway_eff))
            charge_power = max(0, max_charge_power)
            # Update battery state
            P_batt[t] = -charge_power  # charging is negative
            E[t+1] = batt_e + charge_power * oneway_eff * dt
            # Remaining solar after battery charging
            subpanel_net_load_after_charging = subpanel_net_load + charge_power
            # Export from panel up to der_subpanel_export_kw_limit (which is negative)
            subpanel_export_power = min(0,
                                        max(subpanel_net_load_after_charging,
                                            opt_inputs.site_export_kw_limit - main_panel_load,
                                            opt_inputs.der_subpanel_export_kw_limit)
                                        )
            curtailed_solar = abs(subpanel_net_load_after_charging - subpanel_export_power)
            solar_post_curtailment[t] = solar - curtailed_solar
            P_subpanel[t] = subpanel_export_power
            P_grid[t] = main_panel_load + subpanel_export_power
            # Any remaining solar is curtailed
        else:
            # Deficit: try to discharge battery
            max_discharge_power = min(batt_p_max, effective_net_load, (batt_e - e_min) * oneway_eff / dt)
            discharge_power = max(0, max_discharge_power)
            # Update battery state
            P_batt[t] = discharge_power  # discharging is positive
            E[t+1] = batt_e - discharge_power / oneway_eff * dt
            # Remaining deficit after battery discharge
            subpanel_net_load_after_discharging = subpanel_net_load - discharge_power
            if subpanel_net_load_after_discharging > opt_inputs.der_subpanel_import_kw_limit:
                feasible = False
            main_panel_residual_load = main_panel_load + subpanel_net_load_after_discharging

            if main_panel_residual_load > opt_inputs.site_import_kw_limit:
                feasible = False

            if feasible:
                P_grid[t] = main_panel_residual_load
                P_subpanel[t] = subpanel_net_load_after_discharging
                # No curtailment in deficit case
                solar_post_curtailment[t] = solar
            else:
                P_subpanel[t] = np.nan
                P_grid[t] = np.nan
                solar_post_curtailment[t] = np.nan

    res = pd.DataFrame.from_dict({'P_grid': P_grid,
                                  'P_subpanel': P_subpanel,
                                  'P_batt': P_batt,
                                  'E': E[1:],
                                  'solar_post_curtailment': solar_post_curtailment
                                  }).set_index(site_data.index)
    status = 'feasible' if feasible else 'infeasible'
    return OptimizerOutputs(results_df=res, status=status)


def tou_endogenous_sizing_optimization(opt_inputs: OptimizationInputs) -> OptimizerOutputs:
    # Extract parameters from OptimizationInputs
    site_data = opt_inputs.site_data
    batt_rt_eff = opt_inputs.batt_rt_eff
    batt_block_e_max = opt_inputs.batt_block_e_max
    batt_p_max = opt_inputs.batt_block_p_max
    solar_annualized_cost_per_kw = opt_inputs.solar_annualized_cost_per_kw
    batt_annualized_cost_per_unit = opt_inputs.batt_annualized_cost_per_unit
    integer_problem = opt_inputs.integer_problem
    
    """Assumes that solar data in the site_data is per kW"""
    simulation_years = (site_data.index[-1] - site_data.index[0]).total_seconds() / (365 * 24 * 60 * 60)
    tariff = opt_inputs.tariff_model.tariff_timeseries.copy()
    tariff = tariff.loc[site_data.index[0]:site_data.index[-1],:]

    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
    time_intervals = site_data.index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600

    px_sell_dt = tariff['energy_export_rate_kwh'] * dt
    px_buy_dt = tariff['energy_import_rate_kwh'] * dt

    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    n = site_data.shape[0]
    E_transition = sps.hstack([sps.eye(n, format="csr"), sps.csr_matrix((n, 1))], format="csr")
    # E_transition = np.hstack([np.eye(n), np.zeros(n).reshape(-1,1)])
    starting_soe = opt_inputs.batt_starting_soe if opt_inputs.batt_starting_soe is not None else backup_reserve

    s_size_kw = cp.Variable(integer=integer_problem)
    n_batts = cp.Variable(integer=integer_problem)
    batt_e_max = n_batts * batt_block_e_max
    starting_energy_kwh = starting_soe * batt_e_max
    e_min = backup_reserve * batt_e_max
    E_0 = starting_energy_kwh
    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
    P_subpanel_import = cp.Variable(n)
    P_subpanel_export = cp.Variable(n)
    solar_post_curtailment = cp.Variable(n, nonneg=True)
    E = cp.Variable(n+1)

    # Power flows are all AC, and are signed relative to the bus: injections to the bus are positive, withdrawals/exports from the bus are negative

    constraints = [-batt_p_max <= P_batt_charge,
                   P_batt_charge <= 0,
                   0 <= s_size_kw,
                   s_size_kw <= 15,
                   0 <= n_batts,
                   n_batts <= 10,
                   0 <= P_batt_discharge,
                   P_batt_discharge <= batt_p_max,
                   0 <= P_grid_buy,
                   P_grid_sell <= 0,
                   0 <= P_subpanel_import,
                   P_subpanel_import <= opt_inputs.der_subpanel_import_kw_limit,
                   P_subpanel_export <= 0,
                   opt_inputs.der_subpanel_export_kw_limit <= P_subpanel_export,
                   P_grid_buy + P_grid_sell - site_data['main_panel_load'] + P_subpanel_import + P_subpanel_export == 0,
                   e_min <= E,
                   E <= batt_e_max,
                   solar_post_curtailment >= 0,
                   solar_post_curtailment <= cp.Constant(site_data['solar']) * s_size_kw,
                   E[1:] == E_transition @ E - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                   P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - site_data['der_subpanel_load'] - site_data['main_panel_load'] + solar_post_curtailment == 0,
                   E[0] == E_0
                   ]

    obj = cp.Minimize(P_grid_sell @ px_sell_dt +
                      P_grid_buy @ px_buy_dt +
                      n_batts * batt_annualized_cost_per_unit * simulation_years +
                      s_size_kw * solar_annualized_cost_per_kw * simulation_years
                      )

    prob = cp.Problem(obj, constraints)

    opt_start = time.time()
    prob.solve()
    print(f"Optimization done in {time.time() - opt_start :.3f} seconds")

    res = pd.DataFrame.from_dict({'P_batt': P_batt_charge.value + P_batt_discharge.value,
                        'P_grid': P_grid_buy.value + P_grid_sell.value,
                        'P_subpanel': P_subpanel_import.value + P_subpanel_export.value,
                        'E': E[1:].value,
                        'solar_post_curtailment': solar_post_curtailment.value,
                              }).set_index(site_data.index,)
    sizing_results = {'n_batt_blocks': n_batts.value, 'n_solar': s_size_kw.value}
    return OptimizerOutputs(results_df=res, sizing_results=sizing_results)
