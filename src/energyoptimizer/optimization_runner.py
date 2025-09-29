import pandas as pd



class OptimizationInputs:
    """
    Input: OptimizationRunnerInputs
    Structured for optimizer: OptimizationInputs

    """
    site_data : pd.DataFrame
    tariff_model: TariffModel
    batt_rt_eff: float
    batt_block_e_max: float
    batt_block_p_max: float
    backup_reserve: float
    circuit_import_kw_limit: float
    circuit_export_kw_limit: float
    site_import_kw_limit: float
    site_export_kw_limit: float

class OptimizationClock:
    """We may want to create non-overlapping optimization windows (typically for design),
    or we may want rolling optimization with a long forecasting window, but frequent re-optimization (model predictive control archetype).
    """
    frequency: str  # e.g., '1D' for daily, '1H' for hourly
    horizon: str  # e.g., '7D' for 7 days, '1D' for 1 day

class OptimizationRunner:
    """
    Take the OptimizationRunnerInputs
    Structure the site data and other inputs
    if needed, split the data into multiple chunks based on the optimization_freq
    Call the appropriate optimizer
    Stitch together the results (if needed)
    Return the results
    """

