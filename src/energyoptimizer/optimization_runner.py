import pandas as pd
import enum
from typing import Optional, TYPE_CHECKING
import attrs
from .optimizers import tou_optimization, self_consumption, tou_endogenous_sizing_optimization, OptimizationInputs, OptimizerOutputs

if TYPE_CHECKING:
    from .batteryopt_interface import DesignInputs, FinancialModelInputs


class OptimizationType(enum.Enum):
    """Enum for different optimization types available in the system."""
    SELF_CONSUMPTION = "self_consumption"
    TOU_OPTIMIZATION = "tou_optimization"
    TOU_ENDOGENOUS_SIZING = "tou_endogenous_sizing"


# Global mapping of optimization types to their functions
OPTIMIZER_FUNCTIONS = {
    OptimizationType.SELF_CONSUMPTION: self_consumption,
    OptimizationType.TOU_OPTIMIZATION: tou_optimization,
    OptimizationType.TOU_ENDOGENOUS_SIZING: tou_endogenous_sizing_optimization
}


@attrs.define
class OptimizationClock:
    """We may want to create non-overlapping optimization windows (typically for design),
    or we may want rolling optimization with a long forecasting window, but frequent re-optimization (model predictive control archetype).
    """
    frequency: str | pd.DateOffset  # e.g., '1D' for daily, '1H' for hourly
    horizon: Optional[pd.DateOffset] = None  # e.g., '7D' for 7 days, '1D' for 1 day
    lookback: Optional[pd.DateOffset] = None

    def get_intervals(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
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


@attrs.define
class OptimizationRunnerInputs:
    optimization_type: OptimizationType
    optimization_start: pd.Timestamp
    optimization_end: pd.Timestamp
    design_inputs: 'DesignInputs'
    financial_model_inputs: 'FinancialModelInputs'
    optimization_clock: Optional[OptimizationClock] = None
    parallelize: bool = True



class OptimizationRunner:
    """
    Take the OptimizationRunnerInputs
    Structure the site data and other inputs
    if needed, split the data into multiple chunks based on the optimization_freq
    Call the appropriate optimizer
    Stitch together the results (if needed)
    Return the results
    """
    
    def __init__(self, inputs: OptimizationRunnerInputs, design_inputs: 'DesignInputs', 
                 financial_model_inputs: 'FinancialModelInputs'):
        """Initialize the OptimizationRunner with required inputs."""
        self.inputs = inputs
        self.design_inputs = design_inputs
        self.financial_model_inputs = financial_model_inputs
        
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
        tariff_model = self.design_inputs.tariff_model
        
        # Run optimization for each interval
        all_results = []
        all_sizing_results = {}
        
        for optimize_at, data_from, data_until in intervals:
            # Extract data for this interval
            interval_data = site_data.loc[data_from:data_until]
            
            # Create optimization inputs
            opt_inputs = OptimizationInputs(
                site_data=interval_data,
                tariff_model=tariff_model,
                batt_rt_eff=self.design_inputs.batt_rt_eff,
                batt_block_e_max=self.design_inputs.batt_block_e_max,
                batt_block_p_max=self.design_inputs.batt_block_p_max,
                backup_reserve=self.design_inputs.backup_reserve,
                circuit_import_kw_limit=self.design_inputs.circuit_import_kw_limit,
                circuit_export_kw_limit=self.design_inputs.circuit_export_kw_limit,
                site_import_kw_limit=self.design_inputs.site_import_kw_limit,
                site_export_kw_limit=self.design_inputs.site_export_kw_limit
            )
            
            # Run the appropriate optimizer
            optimizer_output = self._run_optimizer(opt_inputs)
            all_results.append(optimizer_output.get_results())
            
            # Collect sizing results if available
            if optimizer_output.get_sizing_results():
                all_sizing_results.update(optimizer_output.get_sizing_results())
        
        # Combine results if multiple intervals
        if len(all_results) == 1:
            combined_results = all_results[0]
        else:
            combined_results = pd.concat(all_results).sort_index()
        
        return OptimizerOutputs(combined_results, all_sizing_results)
    
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
