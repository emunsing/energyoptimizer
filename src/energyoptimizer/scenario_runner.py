import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol, Callable
import attrs
from multiprocessing import Pool
from functools import partial

from .optimizers import OptimizerOutputs
from .batteryopt_interface import DesignInputs, FinancialModelInputs, ScenarioSpec, OptimizationType, OptimizationRunnerInputs, OptimizationClock, GeneralAssumptions, DesignSpec, TariffSpec, FinancialSpec
from .optimization_runner import OptimizationRunner
from .tariff.tariff_utils import TariffModel


def _run_single_optimization(runner_inputs: OptimizationRunnerInputs) -> OptimizerOutputs:
    """
    Helper function to run a single optimization.
    This is used for parallel processing.
    """
    runner = OptimizationRunner(runner_inputs)
    return runner.run_optimization()


def closest_n_elements(x, n):
    """
    Given a 2D point x, return the n closest integer points to x.
    
    Args:
        x: 2D point (solar_size, battery_blocks)
        n: Number of closest points to return
    
    Returns:
        List of tuples (solar_size, battery_blocks) representing the n closest integer points
    """
    assert len(x) == 2
    r = int(np.ceil(np.sqrt(n))) + 1
    
    # Create grid of integer points around x
    xs = np.arange(max(0, int(np.floor(x[0])) - r), int(np.ceil(x[0])) + r + 1)
    ys = np.arange(max(0, int(np.floor(x[1])) - r), int(np.ceil(x[1])) + r + 1)
    grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    # Compute Euclidean distances to all integer points
    distances = np.linalg.norm(grid - x, axis=1)

    # Get indices of n closest points
    idx = np.argsort(distances)[:n]

    return [tuple(map(int, grid[i])) for i in idx]


@attrs.define
class ResultSummary:
    """Container for summarized optimization results."""
    combined_timeseries: pd.DataFrame = None
    billing_cycles: pd.DataFrame = None
    annual_nonfinancial_timeseries: pd.DataFrame = None
    annual_financial_timeseries: pd.DataFrame = None
    summary_stats: pd.Series = None
    optimization_status: str = None
    financial_summary: Dict[str, Any] = None


class ResultSummarizer(Protocol):
    """Protocol for result summarization classes."""
    
    def summarize(self, 
                 optimizer_results: OptimizerOutputs,
                 design_inputs: DesignInputs, 
                 financial_inputs: FinancialModelInputs,
                 tariff_model: TariffModel) -> ResultSummary:
        """
        Summarize optimization results into a standardized format.
        
        Args:
            optimizer_results: Results from the optimization
            design_inputs: Design parameters used
            financial_inputs: Financial model inputs
            tariff_model: Tariff model used
            
        Returns:
            ResultSummary containing summarized data
        """
        ...

class BasicResultSummarizer:
    """Basic implementation of ResultSummarizer that creates comprehensive summaries."""
    
    def summarize(self, 
                 optimizer_results: OptimizerOutputs,
                 design_inputs: DesignInputs, 
                 financial_inputs: FinancialModelInputs,
                 tariff_model: TariffModel) -> ResultSummary:
        """
        Create a comprehensive summary of optimization results.
        """
        results_df = optimizer_results.results_df
        combined_timeseries = self._create_combined_timeseries(results_df, design_inputs, tariff_model)
        billing_cycles = self._compute_billing_cycles(results_df, tariff_model)
        annual_financial_timeseries = self._create_annual_financial_timeseries(optimization_results=optimizer_results,
                                                                               tariff_model=tariff_model,
                                                                               financial_inputs=financial_inputs)
        annual_nonfinancial_timeseries = self._create_annual_nonfinancial_timeseries(results_df=results_df,
                                                                                     design_inputs=design_inputs,
                                                                                     tariff_model=tariff_model)
        summary_stats = self._compute_summary_stats(annual_nonfinancial_timeseries)
        financial_summary = self._extract_financial_summary(financial_inputs, annual_financial_timeseries)
        
        return ResultSummary(
            combined_timeseries=combined_timeseries,
            billing_cycles=billing_cycles,
            annual_nonfinancial_timeseries=annual_nonfinancial_timeseries,
            annual_financial_timeseries=annual_financial_timeseries,
            summary_stats=summary_stats,
            optimization_status=optimizer_results.status,
            financial_summary=financial_summary
        )
    
    def _create_combined_timeseries(self, results_df: pd.DataFrame, 
                                  design_inputs: DesignInputs, 
                                  tariff_model: TariffModel) -> pd.DataFrame:
        """Create combined timeseries from optimizer results, site data, and tariff."""
        # Combine optimizer results with site data
        combined = pd.concat([
            results_df,
            design_inputs.site_data,
            tariff_model.tariff_timeseries
        ], axis=1)
        
        return combined
    
    def _compute_billing_cycles(self, results_df: pd.DataFrame, tariff_model: TariffModel) -> pd.DataFrame:
        """Compute billing cycle data from tariff model."""
        return tariff_model.compute_bill_series(results_df['P_grid'])

    def _create_annual_nonfinancial_timeseries(self, results_df: pd.DataFrame,
                                design_inputs: DesignInputs, 
                                tariff_model: TariffModel,
                                ) -> pd.DataFrame:
        """Create annual timeseries with all required metrics."""
        # Get time step duration
        dt_hours = results_df.index.to_series().diff().dt.total_seconds().iloc[1] / 3600
        
        annual_data = {}
        annual_data['uncurtailed_solar_kwh'] = design_inputs.site_data['solar'] * dt_hours
        
        # Curtailed solar energy production (kWh)
        curtailed_solar = design_inputs.site_data['solar'] - results_df['solar_post_curtailment']
        annual_data['curtailed_solar_kwh'] = curtailed_solar * dt_hours
        
        annual_data['grid_imports_kwh'] = results_df['P_grid'].clip(lower=0) * dt_hours
        annual_data['grid_exports_kwh'] = (-results_df['P_grid'].clip(upper=0)) * dt_hours
        
        # Subpanel imports (kWh) - positive P_subpanel values
        if 'P_subpanel' in results_df.columns:
            annual_data['subpanel_imports_kwh'] = results_df['P_subpanel'].clip(lower=0) * dt_hours
            annual_data['subpanel_exports_kwh'] = (-results_df['P_subpanel'].clip(upper=0)) * dt_hours
        else:
            annual_data['subpanel_imports_kwh'] = pd.Series(0, index=results_df.index)
            annual_data['subpanel_exports_kwh'] = pd.Series(0, index=results_df.index)
        
        # Grid imports by season-period TOU category
        if hasattr(tariff_model, 'tariff_timeseries'):
            tariff_cols = [col for col in tariff_model.tariff_timeseries.columns 
                          if 'season' in col.lower() or 'period' in col.lower() or 'tou' in col.lower()]
            for col in tariff_cols:
                mask = tariff_model.tariff_timeseries[col] == 1
                annual_data[f'grid_imports_{col}_kwh'] = (results_df['P_grid'].clip(lower=0) * mask * dt_hours)

        # Concatenate existent dataframes
        annual_data_df = pd.DataFrame.from_dict(annual_data, orient='columns').resample('1Y').sum()
        return annual_data_df

    def _create_annual_financial_timeseries(self, optimization_results: OptimizerOutputs,
                                           tariff_model: TariffModel,
                                             financial_inputs: FinancialModelInputs
                                           ) -> pd.DataFrame:
        annual_financial_data: list[tuple[str, str], pd.Series] = []

        annual_billing_cycles = self._compute_billing_cycles(results_df, tariff_model).resample('1Y').sum()
        tz = annual_billing_cycles.index.tz
        annual_billing_cycles.loc['category', :] = 'tariff'
        # Add level to column multiindex
        annual_billing_cycles = annual_billing_cycles.T.set_index('category', append=True).T

        annual_cash_flows: list[tuple[str, str], pd.Series] = []
        cash_flow_products = []
        cash_flow_dataframes = []
        for product, cash_flows in financial_inputs.product_cash_flows.items():
            cash_flow_products = cash_flow_products + [product] * cash_flows.unit_cash_flows.shape[1]
            cash_flow_dataframes.append(cash_flows.unit_cash_flows)

        cash_flow_df = pd.concat(cash_flow_dataframes, axis=1)
        cash_flow_df['category'] = cash_flow_products
        cash_flow_df = cash_flow_df.T.set_index('category', append=True).T
        cash_flow_df.index = pd.DatetimeIndex([pd.Timestamp(year=year, month=1, day=1) for year in cash_flow_df.index], tz=tz)

        combined_annual_financials = pd.concat([annual_billing_cycles, cash_flow_df], axis=1)

        return combined_annual_financials
    
    def _compute_summary_stats(self, annual_timeseries: pd.DataFrame) -> pd.Series:
        """Compute average values from annual timeseries."""
        return annual_timeseries.mean()
    
    def _extract_financial_summary(self, financial_inputs: FinancialModelInputs,
                                   annual_financial_timeseries: pd.DataFrame) -> Dict[str, Any]:
        """Extract financial summary information."""
        return {
            'study_years': financial_inputs.study_years,
            'discount_rate': financial_inputs.discount_rate,
            'solar_levelized_unit_cost': financial_inputs.solar_levelized_unit_cost,
            'battery_levelized_unit_cost': financial_inputs.battery_levelized_unit_cost,
            'reference_upgrade_cost': financial_inputs.reference_upgrade_cost
        }


class ScenarioRunner(ABC):
    """Base class for running multiple optimization scenarios."""
    
    def __init__(self, 
                 general_assumptions: GeneralAssumptions,
                 design_spec: DesignSpec,
                 tariff_spec: TariffSpec,
                 financial_spec: FinancialSpec,
                 result_summarizer: ResultSummarizer = None,
                 parallelize: bool = False,
                 n_processes: int = None):
        """
        Initialize the scenario runner.
        
        Args:
            general_assumptions: General assumptions for the study
            design_spec: Design specifications
            tariff_spec: Tariff specifications
            financial_spec: Financial specifications
            result_summarizer: Optional custom result summarizer
            parallelize: Whether to run scenarios in parallel
            n_processes: Number of processes for parallel execution (None = auto)
        """
        self.general_assumptions = general_assumptions
        self.design_spec = design_spec
        self.tariff_spec = tariff_spec
        self.financial_spec = financial_spec
        self.result_summarizer = result_summarizer or BasicResultSummarizer()
        self.parallelize = parallelize
        self.n_processes = n_processes
        
        # Build scenario spec from components
        self.scenario_spec = ScenarioSpec(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec
        )
        
        # Build common inputs
        self.design_inputs = self.scenario_spec.build_design_inputs()
        self.financial_inputs = self.scenario_spec.build_financial_model_inputs()
        self.tariff_model = self.scenario_spec.build_tariff()
        
        # Storage for results
        self.optimizer_results: List[OptimizerOutputs] = []
        self.result_summaries: List[ResultSummary] = []
    
    @abstractmethod
    def _build_runner_inputs_list(self) -> List[OptimizationRunnerInputs]:
        """
        Build list of optimization runner inputs for all scenarios.
        Must be implemented by subclasses.
        """
        pass
    
    def _run_optimizations(self, runner_inputs_list: List[OptimizationRunnerInputs]) -> List[OptimizerOutputs]:
        """
        Run optimizations either sequentially or in parallel.
        
        Args:
            runner_inputs_list: List of optimization runner inputs
            
        Returns:
            List of optimizer results
        """
        if self.parallelize and len(runner_inputs_list) > 1:
            # Run in parallel
            with Pool(processes=self.n_processes) as pool:
                results = pool.map(_run_single_optimization, runner_inputs_list)
        else:
            # Run sequentially
            results = []
            for runner_inputs in runner_inputs_list:
                result = _run_single_optimization(runner_inputs)
                results.append(result)
        
        return results
    
    def run_scenarios(self) -> List[ResultSummary]:
        """
        Run all scenarios and return summarized results.
        """
        # Build list of runner inputs
        runner_inputs_list = self._build_runner_inputs_list()
        
        # Run optimizations (sequentially or in parallel)
        self.optimizer_results = self._run_optimizations(runner_inputs_list)
        
        # Apply result summarizer to each result
        self.result_summaries = []
        for optimizer_result in self.optimizer_results:
            summary = self.result_summarizer.summarize(
                optimizer_result, 
                self.design_inputs, 
                self.financial_inputs, 
                self.tariff_model
            )
            self.result_summaries.append(summary)
        
        return self.result_summaries
    
    def get_result_summaries(self) -> List[ResultSummary]:
        """Get the list of result summaries."""
        return self.result_summaries
    
    def get_optimizer_results(self) -> List[OptimizerOutputs]:
        """Get the list of raw optimizer results."""
        return self.optimizer_results


class SizingSweepScenarioRunner(ScenarioRunner):
    """Scenario runner that performs a Cartesian product sweep of battery and solar sizes."""
    
    def __init__(self, 
                 general_assumptions: GeneralAssumptions,
                 design_spec: DesignSpec,
                 tariff_spec: TariffSpec,
                 financial_spec: FinancialSpec,
                 n_batt_min: int, n_batt_max: int,
                 solar_min: int, solar_max: int,
                 result_summarizer: ResultSummarizer = None,
                 parallelize: bool = False,
                 n_processes: int = None):
        """
        Initialize sizing sweep runner.
        
        Args:
            general_assumptions: General assumptions for the study
            design_spec: Design specifications
            tariff_spec: Tariff specifications
            financial_spec: Financial specifications
            n_batt_min: Minimum number of battery blocks
            n_batt_max: Maximum number of battery blocks
            solar_min: Minimum solar size (kW)
            solar_max: Maximum solar size (kW)
            result_summarizer: Optional custom result summarizer
            parallelize: Whether to run scenarios in parallel
            n_processes: Number of processes for parallel execution (None = auto)
        """
        super().__init__(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec,
            result_summarizer=result_summarizer,
            parallelize=parallelize,
            n_processes=n_processes
        )
        self.n_batt_min = n_batt_min
        self.n_batt_max = n_batt_max
        self.solar_min = solar_min
        self.solar_max = solar_max
    
    def _build_runner_inputs_list(self) -> List[OptimizationRunnerInputs]:
        """
        Build list of optimization runner inputs for all combinations of battery and solar sizes.
        """
        runner_inputs_list = []
        
        # Create Cartesian product of battery and solar sizes
        for n_batt in range(self.n_batt_min, self.n_batt_max + 1):
            for solar_size in range(self.solar_min, self.solar_max + 1):
                # Create modified scenario spec with fixed sizes
                modified_spec = self._create_modified_scenario_spec(n_batt, solar_size)
                
                # Build inputs
                design_inputs = modified_spec.build_design_inputs()
                financial_inputs = modified_spec.build_financial_model_inputs()
                
                runner_inputs = OptimizationRunnerInputs(
                    optimization_type=OptimizationType(self.general_assumptions.optimization_type),
                    optimization_start=self.general_assumptions.start_date,
                    optimization_end=self.general_assumptions.end_date,
                    design_inputs=design_inputs,
                    financial_model_inputs=financial_inputs,
                    optimization_clock=OptimizationClock(frequency='1Y', horizon=None, lookback=None),
                    parallelize=False
                )
                
                runner_inputs_list.append(runner_inputs)
        
        return runner_inputs_list
    
    def _create_modified_scenario_spec(self, n_batt: int, solar_size: int) -> ScenarioSpec:
        """Create a modified scenario spec with fixed sizing parameters."""
        # Create new design spec with fixed sizes
        modified_design_spec = attrs.evolve(
            self.scenario_spec.design_spec,
            min_battery_units=n_batt,
            max_battery_units=n_batt,
            min_solar_units=solar_size,
            max_solar_units=solar_size
        )
        
        # Create new scenario spec
        return attrs.evolve(
            self.scenario_spec,
            design_spec=modified_design_spec
        )


class TopNScenarioRunner(ScenarioRunner):
    """Scenario runner that finds the N closest scenarios to an endogenous sizing result."""
    
    def __init__(self, 
                 general_assumptions: GeneralAssumptions,
                 design_spec: DesignSpec,
                 tariff_spec: TariffSpec,
                 financial_spec: FinancialSpec,
                 n_closest: int,
                 result_summarizer: ResultSummarizer = None,
                 parallelize: bool = False,
                 n_processes: int = None):
        """
        Initialize top-N scenario runner.
        
        Args:
            general_assumptions: General assumptions for the study
            design_spec: Design specifications
            tariff_spec: Tariff specifications
            financial_spec: Financial specifications
            n_closest: Number of closest scenarios to run
            result_summarizer: Optional custom result summarizer
            parallelize: Whether to run scenarios in parallel
            n_processes: Number of processes for parallel execution (None = auto)
        """
        super().__init__(
            general_assumptions=general_assumptions,
            design_spec=design_spec,
            tariff_spec=tariff_spec,
            financial_spec=financial_spec,
            result_summarizer=result_summarizer,
            parallelize=parallelize,
            n_processes=n_processes
        )
        self.n_closest = n_closest
        self.endogenous_result = None
    
    def _build_runner_inputs_list(self) -> List[OptimizationRunnerInputs]:
        """
        Build list of optimization runner inputs for endogenous sizing and N closest scenarios.
        """
        # Step 1: Run endogenous sizing optimization first (this can't be parallelized easily)
        self.endogenous_result = self._run_endogenous_sizing()
        
        # Step 2: Extract optimal sizing
        optimal_sizing = self.endogenous_result.sizing_results
        optimal_point = (optimal_sizing['n_solar'], optimal_sizing['n_batt_blocks'])
        
        # Step 3: Find N closest integer points
        closest_points = closest_n_elements(optimal_point, self.n_closest)
        
        # Step 4: Build runner inputs for closest points (excluding endogenous if it's integer)
        runner_inputs_list = []
        
        for solar_size, n_batt in closest_points:
            if (solar_size, n_batt) != optimal_point:  # Skip if it's the same as endogenous
                modified_spec = self._create_modified_scenario_spec(n_batt, solar_size)
                
                # Build inputs
                design_inputs = modified_spec.build_design_inputs()
                financial_inputs = modified_spec.build_financial_model_inputs()
                
                runner_inputs = OptimizationRunnerInputs(
                    optimization_type=OptimizationType(self.general_assumptions.optimization_type),
                    optimization_start=self.general_assumptions.start_date,
                    optimization_end=self.general_assumptions.end_date,
                    design_inputs=design_inputs,
                    financial_model_inputs=financial_inputs,
                    optimization_clock=OptimizationClock(frequency='1Y', horizon=None, lookback=None),
                    parallelize=False
                )
                
                runner_inputs_list.append(runner_inputs)
        
        return runner_inputs_list
    
    def run_scenarios(self) -> List[ResultSummary]:
        """
        Override run_scenarios to include the endogenous result.
        """
        # Build list of runner inputs (this also runs endogenous sizing)
        runner_inputs_list = self._build_runner_inputs_list()
        
        # Run optimizations for the closest scenarios (sequentially or in parallel)
        closest_results = self._run_optimizations(runner_inputs_list)
        
        # Combine endogenous result with closest results
        self.optimizer_results = [self.endogenous_result] + closest_results
        
        # Apply result summarizer to each result
        self.result_summaries = []
        for optimizer_result in self.optimizer_results:
            summary = self.result_summarizer.summarize(
                optimizer_result, 
                self.design_inputs, 
                self.financial_inputs, 
                self.tariff_model
            )
            self.result_summaries.append(summary)
        
        return self.result_summaries
    
    def _run_endogenous_sizing(self) -> OptimizerOutputs:
        """Run endogenous sizing optimization to find optimal continuous sizing."""
        # Create scenario spec with endogenous sizing enabled
        endogenous_spec = ScenarioSpec(
            general_assumptions=attrs.evolve(
                self.general_assumptions,
                optimization_type='tou_endogenous_sizing'  # Use endogenous sizing type
            ),
            design_spec=self.design_spec,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )
        
        # Build inputs and run optimization
        design_inputs = endogenous_spec.build_design_inputs()
        financial_inputs = endogenous_spec.build_financial_model_inputs()
        
        runner_inputs = OptimizationRunnerInputs(
            optimization_type=OptimizationType.TOU_ENDOGENOUS_SIZING,
            optimization_start=self.general_assumptions.start_date,
            optimization_end=self.general_assumptions.end_date,
            design_inputs=design_inputs,
            financial_model_inputs=financial_inputs,
            optimization_clock=OptimizationClock(frequency='1Y', horizon=None, lookback=None),
            parallelize=False
        )
        
        return _run_single_optimization(runner_inputs)
    
    def _create_modified_scenario_spec(self, n_batt: int, solar_size: int) -> ScenarioSpec:
        """Create a modified scenario spec with fixed sizing parameters."""
        # Create new design spec with fixed sizes
        modified_design_spec = attrs.evolve(
            self.design_spec,
            min_battery_units=n_batt,
            max_battery_units=n_batt,
            min_solar_units=solar_size,
            max_solar_units=solar_size
        )
        
        # Create new scenario spec
        return ScenarioSpec(
            general_assumptions=self.general_assumptions,
            design_spec=modified_design_spec,
            tariff_spec=self.tariff_spec,
            financial_spec=self.financial_spec
        )

