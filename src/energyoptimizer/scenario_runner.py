import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol, Callable
import attrs
import os
import time
from multiprocessing import Pool
from functools import partial

from energyoptimizer.optimizers import OptimizerOutputs, WrappedOptimizerOutputs
from energyoptimizer.batteryopt_interface import (DesignInputs, FinancialModelInputs, ScenarioSpec, OptimizationType,
                                   OptimizationRunnerInputs, OptimizationClock, GeneralAssumptions,
                                   DesignSpec, TariffSpec, FinancialSpec, PRODUCT_TO_SIZING_OUTPUT_MAP)
from energyoptimizer.optimization_runner import OptimizationRunner, OPTIMIZER_CONVENTIONAL_TO_ENDOGENOUS_MAP
from energyoptimizer.batteryopt_utils import SUCCESS_STATUS
from energyoptimizer.tariff.tariff_utils import TariffModel


def _run_single_optimization(runner_inputs: OptimizationRunnerInputs) -> WrappedOptimizerOutputs:
    """
    Helper function to run a single optimization.
    This is used for parallel processing.
    """
    runner = OptimizationRunner(runner_inputs)
    result = runner.run_optimization()
    if result.status not in SUCCESS_STATUS:
        print(f"Warning: Optimization ended with status {result.status}")
        result = WrappedOptimizerOutputs(status=result.status,
                                         design_inputs=runner_inputs.design_inputs,
                                         financial_inputs=runner_inputs.financial_model_inputs)
    return result


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
    sizing_results: Dict[str, Any] = None
    optimization_status: str = None
    financial_summary: Dict[str, Any] = None
    design_inputs: DesignInputs = None
    financial_inputs: FinancialModelInputs = None


class ResultSummarizer(Protocol):
    """Protocol for result summarization classes."""
    
    def summarize(self, 
                 optimizer_results: WrappedOptimizerOutputs,
                  remove_baseline: bool=True) -> ResultSummary:
        """
        Summarize optimization results into a standardized format.
        
        Args:
            optimizer_results: Results from the optimization
            remove_baseline: Whether to remove baseline (no DER) scenario from results
            
        Returns:
            ResultSummary containing summarized data
        """
        ...

class BasicResultSummarizer:
    """Basic implementation of ResultSummarizer that creates comprehensive summaries."""
    
    def summarize(self,
                 optimizer_results: WrappedOptimizerOutputs,
                  remove_baseline=False
                  ) -> ResultSummary:
        """
        Create a comprehensive summary of optimization results.
        """
        design_inputs = optimizer_results.design_inputs
        financial_inputs = optimizer_results.financial_inputs
        tariff_model = optimizer_results.design_inputs.tariff_model
        results_df = optimizer_results.results_df
        sizing_results = optimizer_results.sizing_results
        combined_timeseries = pd.concat([results_df,
                                         design_inputs.site_data,
                                         tariff_model.tariff_timeseries
                                         ], axis=1).loc[results_df.index]
        combined_timeseries['solar'] *= sizing_results.get('n_solar', 0)
        billing_cycles = tariff_model.compute_bill_series(results_df['P_grid'])
        annual_financial_timeseries = self._create_annual_financial_timeseries(optimization_results=optimizer_results)
        annual_nonfinancial_timeseries = self._create_annual_nonfinancial_timeseries(results_df=results_df,
                                                                                     design_inputs=design_inputs,
                                                                                     sizing_results=sizing_results,
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
            financial_summary=financial_summary,
            design_inputs=design_inputs,
            financial_inputs=financial_inputs,
            sizing_results=sizing_results
        )

    def _create_annual_nonfinancial_timeseries(self, results_df: pd.DataFrame,
                                design_inputs: DesignInputs, 
                                tariff_model: TariffModel,
                                               sizing_results: Dict[str, Any]
                                ) -> pd.DataFrame:
        """Create annual timeseries with all required metrics."""
        # Get time step duration
        dt_hours = results_df.index.to_series().diff().dt.total_seconds().iloc[1] / 3600
        
        annual_data = {}
        annual_data['uncurtailed_solar_kwh'] = design_inputs.site_data['solar'] * sizing_results['n_solar'] * dt_hours
        annual_data['solar_post_curtailment'] = results_df['solar_post_curtailment'] * dt_hours
        e_batt_kwh = results_df['P_batt'].clip(lower=0) * dt_hours
        annual_data['batt_dispatch_kwh'] = e_batt_kwh * dt_hours
        batt_internal_energy_delta = e_batt_kwh / design_inputs.batt_rt_eff ** 0.5
        annual_data['battery_cycles'] = batt_internal_energy_delta / (design_inputs.batt_block_e_max * sizing_results['n_batt_blocks'])

        # Curtailed solar energy production (kWh)
        curtailed_solar = design_inputs.site_data['solar'] * sizing_results['n_solar'] - results_df['solar_post_curtailment']
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
        annual_data_df = pd.DataFrame.from_dict(annual_data, orient='columns').resample('1YS').sum()
        return annual_data_df

    @staticmethod
    def _create_annual_financial_timeseries(optimization_results: WrappedOptimizerOutputs,
                                           ) -> pd.DataFrame:
        tariff_model = optimization_results.design_inputs.tariff_model
        financial_inputs = optimization_results.financial_inputs
        results_df = optimization_results.results_df

        billing_cycle_data = tariff_model.compute_bill_series(results_df['P_grid'])
        annual_billing_cycles = billing_cycle_data.resample('1YS').sum()
        annual_billing_cycles.columns.name = 'expense_type'
        tz = annual_billing_cycles.index.tz
        zero_year = annual_billing_cycles.index.year[0]
        annual_billing_cycles = annual_billing_cycles.T
        annual_billing_cycles.loc[:, 'category'] = 'tariff'
        # Add level to column multiindex
        annual_billing_cycles = annual_billing_cycles.set_index('category', append=True)
        annual_billing_cycles = annual_billing_cycles.T

        cash_flow_product_labels = []
        cash_flow_dataframes = []
        for product, cash_flows in financial_inputs.product_cash_flows.items():
            # cash_flow_product_labels = cash_flow_product_labels + [product] * cash_flows.unit_cash_flows.shape[1]
            n_product = optimization_results.sizing_results.get(PRODUCT_TO_SIZING_OUTPUT_MAP[product], 0)
            scaled_cash_flow_df = n_product * cash_flows.unit_cash_flows.copy()
            scaled_cash_flow_df.loc[:, 'annualized_cost'] = cash_flows.unit_annualized_cost * n_product
            cash_flow_product_labels = cash_flow_product_labels + [product] * scaled_cash_flow_df.shape[1]
            cash_flow_dataframes.append(scaled_cash_flow_df)

        cash_flow_df = pd.concat(cash_flow_dataframes, axis=1)
        cash_flow_df.columns.name = 'expense_type'
        cash_flow_df = cash_flow_df.T
        cash_flow_df.loc[:, 'category'] = cash_flow_product_labels
        cash_flow_df = cash_flow_df.set_index('category', append=True)
        cash_flow_df = cash_flow_df.T

        # The cash flow df is zero-indexed by year (0, 1, 2, ...); need to convert to actual years
        assert len(cash_flow_df) == len(annual_billing_cycles)
        cash_flow_df.index = pd.DatetimeIndex([pd.Timestamp(year=year_i + zero_year, month=1, day=1) for year_i in cash_flow_df.index], tz=tz)
        assert cash_flow_df.index.equals(annual_billing_cycles.index)

        combined_annual_financials = pd.concat([annual_billing_cycles, cash_flow_df], axis=1)
        combined_annual_financials.index = pd.DatetimeIndex(combined_annual_financials.index)

        return combined_annual_financials

    @staticmethod
    def _compute_summary_stats(annual_timeseries: pd.DataFrame) -> pd.Series:
        """Compute average values from annual timeseries."""
        return annual_timeseries.mean()

    @staticmethod
    def _extract_financial_summary(financial_inputs: FinancialModelInputs,
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
                 scenario_spec: ScenarioSpec,
                 result_summarizer: ResultSummarizer = None,
                 parallelize: bool = False,
                 n_jobs: int = None):
        """
        Initialize the scenario runner.
        
        Args:
            general_assumptions: General assumptions for the study
            design_spec: Design specifications
            tariff_spec: Tariff specifications
            financial_spec: Financial specifications
            result_summarizer: Optional custom result summarizer
            parallelize: Whether to run scenarios in parallel
            n_jobs: Number of processes for parallel execution (None = auto)
        """
        self.result_summarizer = result_summarizer or BasicResultSummarizer()
        self.parallelize = parallelize
        self.n_jobs = n_jobs
        
        # Build scenario spec from components
        self.scenario_spec = scenario_spec
        
        # Build common inputs
        self.design_inputs = self.scenario_spec.build_design_inputs()
        self.financial_inputs = self.scenario_spec.build_financial_model_inputs()
        self.tariff_model = self.scenario_spec.build_tariff()
        self.optimization_clock = OptimizationClock(frequency=self.scenario_spec.general_assumptions.optimization_clock,
                                                    horizon=self.scenario_spec.general_assumptions.optimization_clock_horizon,
                                                    lookback=self.scenario_spec.general_assumptions.optimization_clock_lookback)
        
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
    
    def _run_optimizations(self, runner_inputs_list: List[OptimizationRunnerInputs]) -> List[WrappedOptimizerOutputs]:
        """
        Run optimizations either sequentially or in parallel.
        
        Args:
            runner_inputs_list: List of optimization runner inputs
            
        Returns:
            List of optimizer results
        """
        if self.parallelize and len(runner_inputs_list) > 1:
            # Run in parallel
            with Pool(processes=self.n_jobs) as pool:
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
        result_cache_file = '/Users/eric/Desktop/temp_results.pkl'
        if os.path.exists(result_cache_file):
            import pickle
            with open(result_cache_file, 'rb') as f:
                self.optimizer_results = pickle.load(f)
        else:
            self.optimizer_results = self._run_optimizations(runner_inputs_list)
            with open(result_cache_file, 'wb') as f:
                import pickle
                pickle.dump(self.optimizer_results, f)

        # self.optimizer_results = self._run_optimizations(runner_inputs_list)
        
        # Apply result summarizer to each result
        self.result_summaries = []
        for optimizer_result in self.optimizer_results:
            if optimizer_result.status not in SUCCESS_STATUS or optimizer_result.results_df is None:
                print(f"Warning: Optimization ended with no results and status {optimizer_result.status}; skipping")
                continue
            summary = self.result_summarizer.summarize(optimizer_result)
            self.result_summaries.append(summary)

        if len(self.result_summaries) == 0:
            print("Warning: No successful optimizations to summarize!")

        return self.result_summaries
    
    def get_result_summaries(self) -> List[ResultSummary]:
        """Get the list of result summaries."""
        return self.result_summaries
    
    def get_optimizer_results(self) -> List[OptimizerOutputs]:
        """Get the list of raw optimizer results."""
        return self.optimizer_results


class SizingSweepScenarioRunner(ScenarioRunner):
    """Scenario runner that performs a Cartesian product sweep of battery and solar sizes."""

    def _build_runner_inputs_list(self) -> List[OptimizationRunnerInputs]:
        """
        Build list of optimization runner inputs for all combinations of battery and solar sizes.
        """
        runner_inputs_list = []
        
        # Create Cartesian product of battery and solar sizes
        for n_batt in range(self.scenario_spec.design_spec.min_battery_units, self.scenario_spec.design_spec.max_battery_units + 1):
            for solar_size in range(self.scenario_spec.design_spec.min_solar_units, self.scenario_spec.design_spec.max_solar_units + 1):
                # Create modified scenario spec with fixed sizes
                modified_design_input = attrs.evolve(
                    self.design_inputs,
                    min_battery_units=n_batt,
                    max_battery_units=n_batt,
                    min_solar_units=solar_size,
                    max_solar_units=solar_size
                )

                # Build inputs
                runner_inputs = OptimizationRunnerInputs(
                    optimization_type=OptimizationType(self.scenario_spec.general_assumptions.optimization_type),
                    optimization_start=self.scenario_spec.general_assumptions.start_date,
                    optimization_end=self.scenario_spec.general_assumptions.end_date,
                    design_inputs=modified_design_input,
                    financial_model_inputs=self.financial_inputs,
                    optimization_clock=self.optimization_clock,
                    parallelize=False
                )
                
                runner_inputs_list.append(runner_inputs)
        
        return runner_inputs_list


class TopNScenarioRunner(ScenarioRunner):
    """Scenario runner that finds the N closest scenarios to an endogenous sizing result."""
    
    def __init__(self, *args, **kwargs):
        """
        Initialize top-N scenario runner.
        Unique args:
            n_closest: Number of closest scenarios to run
        """
        n_closest = kwargs.pop('n_closest', 5)
        respect_bounds = kwargs.pop('respect_bounds', True)
        fixed_size_run_clock = kwargs.pop('optimization_clock', None)  # TopNScenarioRunner Separate clock for fixed-size runs, which are less computationally heavy. By default, the clock is set from the GeneralAssumptions.
        super().__init__(*args, **kwargs)
        self.n_closest = n_closest
        self.endogenous_result = None
        self.respect_bounds = respect_bounds
        if fixed_size_run_clock:
            self.fixed_size_run_clock = fixed_size_run_clock
        else:
            self.fixed_size_run_clock = self.optimization_clock
    
    def _build_runner_inputs_list(self) -> List[OptimizationRunnerInputs]:
        """
        Build list of optimization runner inputs for endogenous sizing and N closest scenarios.
        """
        # Step 1: Run endogenous sizing optimization first (this can't be parallelized easily)
        start_time = time.time()

        result_cache_file = '/Users/eric/Desktop/temp_endogenous_results.pkl'
        if os.path.exists(result_cache_file):
            import pickle
            with open(result_cache_file, 'rb') as f:
                self.endogenous_result = pickle.load(f)
        else:
            self.endogenous_result = self._run_endogenous_sizing()
            with open(result_cache_file, 'wb') as f:
                import pickle
                pickle.dump(self.endogenous_result, f)

        # self.endogenous_result = self._run_endogenous_sizing()
        print("Endogenous sizing completed in {:.2f} seconds".format(time.time() - start_time))

        # Step 2: Extract optimal sizing
        optimal_sizing = self.endogenous_result.sizing_results
        print("Endogenous sizing optimal sizes: ", optimal_sizing)
        optimal_point = (optimal_sizing['n_solar'], optimal_sizing['n_batt_blocks'])
        
        # Step 3: Find N closest integer points
        if self.respect_bounds:
            batt_size_candidates = range(self.scenario_spec.design_spec.min_battery_units, self.scenario_spec.design_spec.max_battery_units + 1)
            solar_size_candidates = range(self.scenario_spec.design_spec.min_solar_units, self.scenario_spec.design_spec.max_solar_units + 1)
            candidate_points = np.array([[solar, batt] for solar in solar_size_candidates for batt in batt_size_candidates])
            distances = np.linalg.norm(candidate_points - optimal_point, axis=1)
            idx = np.argsort(distances)[:self.n_closest]
            closest_points = candidate_points[idx].tolist()
        else:
            closest_points = closest_n_elements(optimal_point, self.n_closest)

        # Step 4: Build runner inputs for closest points (excluding endogenous if it's integer)
        runner_inputs_list = []
        
        for solar_size, n_batt in closest_points:
            modified_design_input = attrs.evolve(
                self.design_inputs,
                min_battery_units=n_batt,
                max_battery_units=n_batt,
                min_solar_units=solar_size,
                max_solar_units=solar_size
            )

            # Build inputs
            runner_inputs = OptimizationRunnerInputs(
                optimization_type=OptimizationType(self.scenario_spec.general_assumptions.optimization_type),
                optimization_start=self.scenario_spec.general_assumptions.start_date,
                optimization_end=self.scenario_spec.general_assumptions.end_date,
                design_inputs=modified_design_input,
                financial_model_inputs=self.financial_inputs,
                optimization_clock=self.fixed_size_run_clock,
                parallelize=False
            )
            runner_inputs_list.append(runner_inputs)
        return runner_inputs_list
    
    def _run_endogenous_sizing(self) -> OptimizerOutputs:
        """Run endogenous sizing optimization to find optimal continuous sizing."""
        # Create scenario spec with endogenous sizing enabled
        baseline_optimization_type = self.scenario_spec.general_assumptions.optimization_type
        assert baseline_optimization_type in OPTIMIZER_CONVENTIONAL_TO_ENDOGENOUS_MAP, f"Use TopNScenarioRunner with an optimization_type in {OPTIMIZER_CONVENTIONAL_TO_ENDOGENOUS_MAP.keys()}"
        updated_optimization_type = OPTIMIZER_CONVENTIONAL_TO_ENDOGENOUS_MAP[baseline_optimization_type]

        runner_inputs = OptimizationRunnerInputs(
            optimization_type=OptimizationType(updated_optimization_type),
            optimization_start=self.scenario_spec.general_assumptions.start_date,
            optimization_end=self.scenario_spec.general_assumptions.end_date,
            design_inputs=self.design_inputs,
            financial_model_inputs=self.financial_inputs,
            optimization_clock=self.optimization_clock,
            parallelize=True,
            n_jobs=self.n_jobs
        )

        return _run_single_optimization(runner_inputs)
