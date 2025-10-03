import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.energyoptimizer.optimizers import OptimizationInputs, subpanel_self_consumption
from src.energyoptimizer.tariff.tariff_utils import TariffModel


class TestSubpanelSelfConsumption:
    """Test cases for subpanel_self_consumption function using predefined test cases."""
    
    @pytest.fixture
    def cases_df(self):
        """Load test cases from CSV file."""
        test_data_path = Path(__file__).parent / "test_data" / "subpanel_self_consumption_test_cases.csv"
        return pd.read_csv(test_data_path).dropna(how='all')
    
    @pytest.fixture
    def mock_tariff_model(self):
        """Create a mock tariff model for testing."""
        # Create a simple mock that has the required tariff_timeseries attribute
        class MockTariffModel:
            def __init__(self):
                # Create a minimal tariff timeseries for a single timestep
                self.tariff_timeseries = pd.DataFrame({
                    'energy_import_rate_kwh': [0.25],  # $0.25/kWh
                    'energy_export_rate_kwh': [0.10]   # $0.10/kWh
                }, index=pd.DatetimeIndex(['2025-01-01 12:00:00'], tz='US/Pacific'))
        
        return MockTariffModel()
    
    def create_optimization_inputs_from_test_case(self, test_case, mock_tariff_model):
        """Create OptimizationInputs from a single test case row."""
        # Create single-timestep site data
        timestamp = pd.DatetimeIndex(['2025-01-01 12:00:00'], tz='US/Pacific')
        
        site_data = pd.DataFrame({
            'main_panel_load': [test_case['main_panel_load']],
            'der_subpanel_load': [test_case['der_subpanel_load']],
            'solar': [test_case['solar']]
        }, index=timestamp)
        
        # Create OptimizationInputs with parameters from test case
        opt_inputs = OptimizationInputs(
            site_data=site_data,
            tariff_model=mock_tariff_model,
            batt_rt_eff=1.00,  # Fixed for all test cases
            batt_block_e_max=test_case['batt_block_e_max'],
            batt_block_p_max=test_case['batt_block_e_max'],  # Fixed for all test cases
            backup_reserve=0.0,  # Fixed for all test cases
            batt_starting_soe=test_case['batt_starting_soe'],
            der_subpanel_import_kw_limit=test_case['der_subpanel_import_kw_limit'],
            der_subpanel_export_kw_limit=test_case['der_subpanel_export_kw_limit'],
            site_import_kw_limit=test_case['site_import_kw_limit'],
            site_export_kw_limit=test_case['site_export_kw_limit'],
            solar_annualized_cost_per_kw=0.15,
            batt_annualized_cost_per_unit=1000.0,
            integer_problem=False
        )
        
        return opt_inputs
    
    def test_subpanel_self_consumption_all_cases(self, cases_df, mock_tariff_model):
        """Test subpanel_self_consumption against all predefined test cases."""

        for idx, test_case in cases_df.iterrows():
                
            # Create optimization inputs from test case
            opt_inputs = self.create_optimization_inputs_from_test_case(test_case, mock_tariff_model)
            
            # Run the optimization
            result = subpanel_self_consumption(opt_inputs)
            
            # Extract results for this single timestep
            results_df = result.results_df
            
            # Check expected status
            expected_status = test_case['status']
            actual_status = result.status
            assert actual_status == expected_status, f"Test case {idx}: Expected status '{expected_status}', got '{actual_status}'"
            
            if actual_status == 'infeasible':
                assert results_df.isnull().sum().sum() > 0, f"Test case {idx}: Expected all null values in infeasible case"
                continue
            
            if actual_status == 'feasible':
                assert results_df.isnull().sum().sum() == 0, f"Test case {idx}: Expected no null values in feasible case"
            # For feasible cases, check values (allowing for small numerical differences)
            tolerance = 1e-6
            
            # Check P_grid
            for col in ['P_grid', 'P_subpanel', 'P_batt', 'E', 'solar_post_curtailment']:
                expected = test_case[col]
                actual = results_df[col].iloc[0]
                assert abs(expected - actual) < tolerance, f"Test case {idx}: {col} expected {expected}, got {actual}"
    
if __name__ == "__main__":
    pytest.main([__file__])
