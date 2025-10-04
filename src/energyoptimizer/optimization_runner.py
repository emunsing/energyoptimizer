import pandas as pd
import enum
from typing import Optional, TYPE_CHECKING
import attrs
from .optimizers import (OptimizationInputs, OptimizerOutputs,
                         subpanel_self_consumption, single_panel_self_consumption,
                         tou_optimization, demand_charge_tou_optimization,
                         tou_endogenous_sizing_optimization, demand_charge_tou_endogenous_sizing_optimization,
                         )

from .batteryopt_interface import DesignInputs, FinancialModelInputs, OptimizationType, OptimizationClock, OptimizationRunnerInputs

from src.energyoptimizer.batteryopt_utils import MIN_DT


# Global mapping of optimization types to their functions
OPTIMIZER_FUNCTIONS = {
    OptimizationType.SIMPLE_SELF_CONSUMPTION: single_panel_self_consumption,
    OptimizationType.SUBPANEL_SELF_CONSUMPTION: subpanel_self_consumption,
    OptimizationType.TOU_OPTIMIZATION: tou_optimization,
    OptimizationType.DEMAND_CHARGE_TOU_OPTIMIZATION: demand_charge_tou_optimization,
    OptimizationType.TOU_ENDOGENOUS_SIZING: tou_endogenous_sizing_optimization,
    OptimizationType.DEMAND_CHARGE_TOU_SIZING_OPTIMIZATION: demand_charge_tou_endogenous_sizing_optimization,
}


class OptimizationRunner:
    """
    Take the OptimizationRunnerInputs
    Structure the site data and other inputs
    if needed, split the data into multiple chunks based on the optimization_freq
    Call the appropriate optimizer
    Stitch together the results (if needed)
    Return the results
    """
    
    def __init__(self, inputs: OptimizationRunnerInputs):
        """Initialize the OptimizationRunner with required inputs."""
        self.inputs = inputs
        self.financial_model_inputs: FinancialModelInputs = inputs.financial_model_inputs
        self.design_inputs: DesignInputs = inputs.design_inputs
        
    def run_optimization(self) -> OptimizerOutputs:
        """
        Run the optimization based on the configured inputs.
        
        Returns:
            OptimizerOutputs containing the optimization results
        """
        # Create optimization clock if needed
        if self.inputs.optimization_clock:
            clock = self.inputs.optimization_clock
            intervals = clock.get_intervals(self.inputs.optimization_start, self.inputs.optimization_end)
        else:
            # Single optimization window
            intervals = [(self.inputs.optimization_start, self.inputs.optimization_start, self.inputs.optimization_end)]
        
        # Prepare site data from design inputs
        site_data = self._prepare_site_data()
        
        # Prepare tariff model
        tariff_model = self.inputs.design_inputs.tariff_model
        
        # Run optimization for each interval
        optimization_results = OptimizerOutputs()
        
        for optimize_at, data_from, data_until in intervals:
            # Extract data for this interval
            full_visible_interval_data = site_data.loc[data_from:data_until - MIN_DT]

            ### IMPORTANT NOTE: ####
            # Any forecasting or pre-processing which requires lookback/horizon info should be done here if
            # After this, all data should be aligned to the optimization index, and reflect the point-in-time knowledge at optimize_at
            ########################

            # At this point, all data should be aligned to the optimization index, and reflect the point-in-time knowledge at optimize_at
            interval_data = site_data.loc[optimize_at:data_until - MIN_DT]

            # Create optimization inputs
            opt_inputs = OptimizationInputs(
                start=optimize_at,
                end=data_until,
                site_data=interval_data,
                tariff_model=tariff_model,
                batt_rt_eff=self.design_inputs.batt_rt_eff,
                batt_block_e_max=self.design_inputs.batt_block_e_max,
                batt_block_p_max=self.design_inputs.batt_block_p_max,
                backup_reserve=self.design_inputs.backup_reserve,
                der_subpanel_import_kw_limit=self.design_inputs.der_subpanel_import_kw_limit,
                der_subpanel_export_kw_limit=self.design_inputs.der_subpanel_export_kw_limit,
                site_import_kw_limit=self.design_inputs.site_import_kw_limit,
                site_export_kw_limit=self.design_inputs.site_export_kw_limit
            )
            
            # Run the appropriate optimizer
            optimizer_output = self._run_optimizer(opt_inputs)
            optimization_results.append(optimizer_output)

        # Combine all results and trim to the optimization period (results could be longer if `horizon` > 0)
        optimization_results.finalize(self.inputs.optimization_start, self.inputs.optimization_end)

        return optimization_results
    
    def _prepare_site_data(self) -> pd.DataFrame:
        """Prepare site data from design inputs."""
        # This would extract and format the site data from design_inputs
        # For now, assuming the design_inputs has the necessary data structure
        return self.design_inputs.site_data
    
    def _run_optimizer(self, opt_inputs: OptimizationInputs) -> OptimizerOutputs:
        """Run the appropriate optimizer based on optimization type."""
        optimizer_func = OPTIMIZER_FUNCTIONS.get(self.inputs.optimization_type)
        if optimizer_func is None:
            raise ValueError(f"Unknown optimization type: {self.inputs.optimization_type}")
        
        # Run the optimizer - now all optimizers return OptimizerOutputs
        return optimizer_func(opt_inputs)
