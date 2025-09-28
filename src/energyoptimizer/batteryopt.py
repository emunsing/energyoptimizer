import cvxpy as cp
import pandas as pd
import numpy as np
import time

def merge_solar_and_load_data(elec_usage: pd.Series, solar_ac_estimate: pd.Series, timezone='US/Pacific', verbose=False) -> pd.DataFrame:
    """Because pvlib uses solar data which is in the past, we need to shift the solar data to match
    the actual electricity usage data.
    Right now, this works one year at a time, and relies on a reference set of years hard-coded to be 2019-2021, around the 2020 leap year.
    """
    assert len(elec_usage.index.year.unique()) == 1, "Electricity usage data must be for a single year"
    elec_end_date = elec_usage.index[-1]
    leap_day = None
    for t in elec_usage.index:
        if (t.month==2) & (t.day==29):
            leap_day = t
            break

    if solar_ac_estimate.index[0] <= elec_usage.index[0] and (solar_ac_estimate.index[-1] + pd.DateOffset(hours=1)) >= elec_usage.index[-1]:
        shift_by_yrs = 0
    elif leap_day:
        shift_by_yrs = 2020 - leap_day.year
    elif elec_end_date.month == 12 and elec_end_date.day == 31:
        shift_by_yrs = 2019 - elec_end_date.year
    elif elec_end_date.month > 2:
        shift_by_yrs = 2021 - elec_end_date.year
    else:
        shift_by_yrs = 2019 - elec_end_date.year

    if verbose:
        print("solar_ac_estimate")
        print(solar_ac_estimate.head())
        print("elec_usage")
        print(elec_usage.head())

    solar_ac_estimate.index = (solar_ac_estimate.index.tz_convert('UTC') - pd.DateOffset(years=shift_by_yrs)).tz_convert(timezone)
    solar_ac_estimate = solar_ac_estimate.resample('1h', closed='right').last().ffill()  # Deal with any gaps related to shifted DST; thankfully these are in the night

    site_data = pd.DataFrame(elec_usage).join(solar_ac_estimate, how='left').tz_convert(timezone)
    site_data['solar'] = site_data['solar'].interpolate()
    return site_data


def run_optimization(site_data: pd.DataFrame,
                     tariff: pd.DataFrame,
                     batt_rt_eff=0.85,
                     batt_e_max=13.5,
                     batt_p_max=5,
                     backup_reserve=0.2,
                     import_kw_limit = 100,
                     export_kw_limit=-100) -> pd.DataFrame:
    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
    time_intervals = site_data.index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600

    oneway_eff = np.sqrt(batt_rt_eff)
    e_min = backup_reserve * batt_e_max

    n = site_data.shape[0]
    E_0 = e_min

    E_transition = np.hstack([np.eye(n), np.zeros(n).reshape(-1,1)])

    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
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
                e_min <= E,
                E <= batt_e_max,
                solar_post_curtailment >= 0,
                solar_post_curtailment <= site_data['solar'],
                E[1:] == E_transition @ E - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - site_data['load'] + solar_post_curtailment == 0,
                E[0] == E_0
                ]

    obj = cp.Minimize(P_grid_sell @ tariff['px_sell'] + P_grid_buy @ tariff['px_buy'])

    prob = cp.Problem(obj, constraints)

    opt_start = time.time()
    prob.solve()
    if prob.status == cp.INFEASIBLE:
        raise AssertionError("Infeasible")
    print(f"Optimization done in {time.time() - opt_start :.3f} seconds")

    res = pd.DataFrame.from_dict({'P_batt': P_batt_charge.value + P_batt_discharge.value,
                                  'P_grid': P_grid_buy.value + P_grid_sell.value,
                                  'E': E[1:].value,
                                  'solar_post_curtailment': solar_post_curtailment.value
                                  }).set_index(site_data.index)
    return res

def run_self_consumption(site_data: pd.DataFrame,
                        tariff: pd.DataFrame,
                        batt_rt_eff=0.85,
                        batt_e_max=13.5,
                        batt_p_max=5,
                        backup_reserve=0.2,
                        import_kw_limit=100,
                        export_kw_limit=-100) -> pd.DataFrame:
    """
    Simple self-consumption battery dispatch algorithm.
    Returns DataFrame with columns: P_batt, P_grid, E, solar_post_curtailment
    """
    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"
    time_intervals = site_data.index.diff()[1:].unique()
    assert len(time_intervals) == 1, "Dataframes must have a constant-interval time index"
    dt = time_intervals[0].total_seconds() / 3600

    oneway_eff = np.sqrt(batt_rt_eff)
    e_min = backup_reserve * batt_e_max
    n = site_data.shape[0]
    E = np.zeros(n+1)
    E[0] = e_min
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
    return res

def run_endogenous_sizing_optimization(site_data: pd.DataFrame,
                     tariff: pd.DataFrame,
                     solar_annualized_cost_per_kw=3.0 / 20,
                     batt_annualized_cost_per_unit=1000,
                     batt_rt_eff=0.85,
                     batt_block_e_max=13.5,
                     batt_p_max=5,
                                       integer_problem=False,
                     ) -> tuple[float, float, pd.DataFrame]:
    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"

    """Assumes that solar data in the site_data is per kW"""
    simulation_years = (site_data.index[-1] - site_data.index[0]).total_seconds() / (365 * 24 * 60 * 60)

    dt = 1.0

    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    n = site_data.shape[0]
    E_transition = np.hstack([np.eye(n), np.zeros(n).reshape(-1,1)])

    s_size_kw = cp.Variable(integer=integer_problem)
    n_batts = cp.Variable(integer=integer_problem)
    batt_e_max = n_batts * batt_block_e_max
    e_min = backup_reserve * batt_e_max
    E_0 = e_min
    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
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
                   e_min <= E,
                   E <= batt_e_max,
                   E[1:] == E_transition @ E - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                   P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - site_data['load'] + s_size_kw * site_data['solar'] == 0,
                   E[0] == E_0
                   ]

    obj = cp.Minimize(P_grid_sell @ tariff['px_sell'] +
                      P_grid_buy @ tariff['px_buy'] +
                      n_batts * batt_annualized_cost_per_unit * simulation_years +
                      s_size_kw * solar_annualized_cost_per_kw * simulation_years
                      )

    prob = cp.Problem(obj, constraints)

    opt_start = time.time()
    prob.solve()
    print(f"Optimization done in {time.time() - opt_start :.3f} seconds")

    res = pd.DataFrame.from_dict({'P_batt': P_batt_charge.value + P_batt_discharge.value,
                        'P_grid': P_grid_buy.value + P_grid_sell.value,
                        'E': E[1:].value}).set_index(site_data.index)
    return n_batts.value, s_size_kw.value, res
