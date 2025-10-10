"""
Sales Engineering Solar Sizing Application
===========================================

This module provides a Panel-based web application for solar + battery system sizing
using the energyoptimizer backend. It demonstrates the extensible architecture where:
- Domain-specific WebInputs/WebOutputs classes encapsulate the business logic
- UI components are built from these domain classes
- Backend integration is clean and type-safe
"""

import panel as pn
import attrs
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path

from energyoptimizer.batteryopt_interface import ScenarioSpec
from energyoptimizer.scenario_runner import ResultSummary, BasicResultSummarizer
from weboptimizer.sales_engineering_backend import ScenarioStudy

pn.extension(sizing_mode="stretch_width", notifications=True)


# ============================================================================
# Domain-Specific Input/Output Classes
# ============================================================================

@attrs.define
class SalesEngineeringInputs:
    """
    Web inputs for sales engineering application.
    Maps directly to ScenarioStudy initialization parameters.
    """
    id: int | None = None
    scenario_name: str = ""

    # Study parameters
    start_date: str = '2026-01-01'
    study_years: int = 2
    optimize_sizing: bool = True  # If True, run sizing_optimizer; else sizing_sweep

    # DataFrames (required uploads)
    unit_solar_timeseries_kw: pd.DataFrame | None = None
    der_subpanel_load_kw: pd.DataFrame | None = None
    main_panel_load_kw: pd.DataFrame | None = None

    # Site parameters
    available_circuit_capacity_amps: int = 100
    site_max_capacity_amps: int = 400
    site_allows_export: bool = False
    panel_voltage: int = 480

    # Battery specifications
    battery_unit_size: float = 0.55 * 210  # kWh, sized at end of life
    battery_unit_power: float = 60  # kW
    min_battery_units: int = 0
    max_battery_units: int = 5

    # Solar specifications
    min_solar_units: int = 0
    max_solar_units: int = 30

    # Financial assumptions
    solar_capital_cost_per_unit: float = 156e3
    battery_capital_cost_per_unit: float = 89444 + 5000 + 11441 + 38000
    solar_lifetime: int = 20
    solar_residual_value_end_of_life: float = 0.0
    battery_lifetime: int = 20
    battery_residual_value_end_of_life: float = 0.0
    discount_rate: float = 0.00
    itc_rate: float = 0.30

    # Tariff assumptions
    annual_rate_escalator: float = 0.06

    def validate_dataframes(self) -> tuple[bool, str]:
        """Validate that required DataFrames are present."""
        missing = []
        if self.unit_solar_timeseries_kw is None:
            missing.append("Solar timeseries")
        if self.der_subpanel_load_kw is None:
            missing.append("DER subpanel load")
        if self.main_panel_load_kw is None:
            missing.append("Main panel load")

        if missing:
            return False, f"Missing required data: {', '.join(missing)}"
        return True, "OK"
    
    def validate_scenario_count(self) -> tuple[bool, str]:
        """Validate that the number of scenarios is reasonable."""
        if not self.optimize_sizing:
            n_battery_scenarios = (self.max_battery_units - self.min_battery_units) + 1
            n_solar_scenarios = (self.max_solar_units - self.min_solar_units) + 1
            total_scenarios = n_battery_scenarios * n_solar_scenarios
            if total_scenarios > 10:
                return False, f"Too many scenarios ({total_scenarios}). Please reduce the range to generate <= 10 scenarios."
        return True, "OK"

    def to_scenario_study(self) -> ScenarioStudy:
        """Convert WebInputs to ScenarioStudy backend object."""
        # Create the scenario study with only the attrs-defined fields
        scenario_study = ScenarioStudy(
            start_date=self.start_date,
            study_years=self.study_years,
            # Timeseries
            unit_solar_timeseries_kw=self.unit_solar_timeseries_kw,
            der_subpanel_load_kw=self.der_subpanel_load_kw,
            main_panel_load_kw=self.main_panel_load_kw,
            # Site specs
            available_circuit_capacity_amps=self.available_circuit_capacity_amps,
            site_max_capacity_amps=self.site_max_capacity_amps,
            site_allows_export=self.site_allows_export,
            panel_voltage=self.panel_voltage,
            # System sizing
            battery_unit_size=self.battery_unit_size,
            battery_unit_power=self.battery_unit_power,
            min_battery_units=self.min_battery_units,
            max_battery_units=self.max_battery_units,
            min_solar_units=self.min_solar_units,
            max_solar_units=self.max_solar_units,
            # Financials
            solar_capital_cost_per_unit=self.solar_capital_cost_per_unit,
            battery_capital_cost_per_unit=self.battery_capital_cost_per_unit,
            solar_lifetime=self.solar_lifetime,
            battery_lifetime=self.battery_lifetime,
            solar_residual_value_end_of_life=self.solar_residual_value_end_of_life,
            battery_residual_value_end_of_life=self.battery_residual_value_end_of_life,
            discount_rate=self.discount_rate,
            itc_rate=self.itc_rate,
        )
        return scenario_study

    def compute(self) -> list['SalesEngineeringOutputs']:
        """
        Run the computation and return a list of outputs.
        This encapsulates the logic of single vs multiple scenarios.
        """
        # Validate inputs
        valid, msg = self.validate_dataframes()
        if not valid:
            raise ValueError(msg)
        
        # Validate scenario count
        valid, msg = self.validate_scenario_count()
        if not valid:
            raise ValueError(msg)

        # Create ScenarioStudy
        scenario_study = self.to_scenario_study()

        # Run appropriate computation
        if self.optimize_sizing:
            results = scenario_study.sizing_optimizer()
        else:
            results = scenario_study.sizing_sweep()

        # Convert results to outputs
        outputs = []
        for i, result in enumerate(results):  # these are ResultSummary objects
            output = SalesEngineeringOutputs.from_result_summary(
                result,
                scenario_id=self.id if i == 0 else None
            )
            outputs.append(output)

        return outputs


@attrs.define
class SalesEngineeringOutputs:
    """
    Web outputs for sales engineering application.
    Stores ResultSummary data in a format suitable for display.
    """
    id: int | None = None
    scenario_name: str = ""
    summary_stats: pd.Series | None = None
    sizing_results: dict | None = None
    optimization_status: str | None = None
    financial_summary: dict | None = None

    @classmethod
    def from_result_summary(cls, result: ResultSummary, scenario_id: int | None = None, scenario_name: str = "") -> 'SalesEngineeringOutputs':
        """Create WebOutputs from a ResultSummary object."""
        # Auto-generate name from sizing results if not provided
        if not scenario_name and result.sizing_results:
            n_batt = result.sizing_results.get('n_batt_blocks', 0)
            n_solar = result.sizing_results.get('n_solar', 0)
            scenario_name = f"{n_batt} batts, {n_solar} solar"
        
        return cls(
            id=scenario_id,
            scenario_name=scenario_name,
            summary_stats=result.summary_stats,
            sizing_results=result.sizing_results,
            optimization_status=result.optimization_status,
            financial_summary=result.financial_summary,
        )

    def render_summary_table(self) -> pd.DataFrame:
        """Render summary stats as a DataFrame for display."""
        if self.summary_stats is None:
            return pd.DataFrame({"Message": ["No results available"]})
        results_to_show = pd.concat([self.summary_stats.copy(),
                                     pd.Series(self.sizing_results)
                                     ])
        return pd.DataFrame(results_to_show)

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "id": self.id,
            "optimization_status": self.optimization_status,
            "sizing": self.sizing_results,
            "stats": self.summary_stats.to_dict() if self.summary_stats is not None else None
        }


# ============================================================================
# Helper Functions for DataFrame Loading
# ============================================================================

def load_dataframe_from_csv(file_contents: bytes, expected_columns: list[str]) -> pd.DataFrame:
    """
    Load a DataFrame from CSV bytes and validate it has expected columns.

    Args:
        file_contents: Raw bytes from file upload
        expected_columns: List of acceptable column names

    Returns:
        DataFrame with validated columns
    """
    import io
    df = pd.read_csv(io.BytesIO(file_contents))

    # Check if any expected column is present
    found_cols = [col for col in expected_columns if col in df.columns]
    if not found_cols:
        raise ValueError(f"CSV must contain one of these columns: {expected_columns}")

    return df


# ============================================================================
# Web Scenario Class (similar to MVP)
# ============================================================================

@attrs.define
class SalesEngineeringScenario:
    """Container for inputs and outputs of a single scenario."""
    id: int
    inputs: SalesEngineeringInputs
    outputs: SalesEngineeringOutputs

    def update_id(self, new_id: int) -> 'SalesEngineeringScenario':
        """Create a new scenario with updated ID."""
        updated_inputs = attrs.evolve(self.inputs, id=new_id)
        updated_outputs = attrs.evolve(self.outputs, id=new_id)
        return attrs.evolve(self, id=new_id, inputs=updated_inputs, outputs=updated_outputs)


# ============================================================================
# Consolidated Results Display
# ============================================================================

def show_consolidated_results(all_outputs: list[SalesEngineeringOutputs]) -> pd.DataFrame:
    """Generate a consolidated view of all scenario results."""
    if not all_outputs:
        return pd.DataFrame({"Message": ["No results available"]})

    data = []
    for output in all_outputs:
        row = {
            "ID": output.id,
            "Name": output.scenario_name or "Unnamed",
            "Status": output.optimization_status or "N/A"
        }
        
        if output.sizing_results:
            row["Solar Units"] = output.sizing_results.get('n_solar', 'N/A')
            row["Battery Blocks"] = output.sizing_results.get('n_batt_blocks', 'N/A')
        else:
            row["Solar Units"] = "N/A"
            row["Battery Blocks"] = "N/A"
        
        if output.summary_stats is not None and 'grid_imports_kwh' in output.summary_stats:
            row["Grid Imports (kWh/yr)"] = f"{output.summary_stats['grid_imports_kwh']:,.0f}"
        else:
            row["Grid Imports (kWh/yr)"] = "N/A"
        
        data.append(row)

    return pd.DataFrame(data)


# ============================================================================
# UI Factory Function
# ============================================================================

def create_scenario_ui(scenario_idx: int, scenario: SalesEngineeringScenario,
                       scenarios_registry: dict, scenario_outputs_registry: dict,
                       scenarios_container, consolidated_results_display,
                       refresh_callback, add_callback):
    """
    Create the UI for a single scenario.
    This factory function creates all widgets and callbacks for one scenario instance.
    """
    inputs = scenario.inputs
    outputs = scenario.outputs

    # ===== Input Widgets =====

    # Scenario name
    scenario_name_input = pn.widgets.TextInput(name="Scenario Name", value=inputs.scenario_name or f"Scenario {scenario.id}", width=250)

    # Study parameters section
    study_params_header = pn.pane.Markdown("### Study Parameters")

    start_date_input = pn.widgets.TextInput(name="Start Date", value=inputs.start_date, width=120)
    study_years_input = pn.widgets.IntInput(name="Study Years", value=inputs.study_years, start=1, end=20, width=100)
    optimize_checkbox = pn.widgets.Checkbox(name="Optimize Sizing", value=inputs.optimize_sizing, width=150)

    # Sizing parameters section
    sizing_header = pn.pane.Markdown("### Sizing Parameters")
    min_battery_input = pn.widgets.IntInput(name="Battery Units: Min", value=inputs.min_battery_units, start=0,
                                            width=100)
    max_battery_input = pn.widgets.IntInput(name="Max", value=inputs.max_battery_units, start=0, width=100,
                                            disabled=not inputs.optimize_sizing)
    min_solar_input = pn.widgets.IntInput(name="Solar units: Min", value=inputs.min_solar_units, start=0, width=100)
    max_solar_input = pn.widgets.IntInput(name="Max", value=inputs.max_solar_units, start=0, width=100,
                                          disabled=not inputs.optimize_sizing)

    # File upload widgets
    uploads_header = pn.pane.Markdown("### Data Uploads")
    solar_upload = pn.widgets.FileInput(name="Solar Timeseries CSV", accept=".csv", width=250)
    der_load_upload = pn.widgets.FileInput(name="DER Subpanel Load CSV", accept=".csv", width=250)
    main_load_upload = pn.widgets.FileInput(name="Main Panel Load CSV", accept=".csv", width=250)
    load_example_btn = pn.widgets.Button(name="Load Example Data", button_type="warning", width=150)

    # Upload status indicators
    solar_status = pn.widgets.StaticText(
        name="Solar Data",
        value="✓ Loaded" if inputs.unit_solar_timeseries_kw is not None else "⚠ Not loaded",
        width=200
    )
    der_load_status = pn.widgets.StaticText(
        name="DER Load Data",
        value="✓ Loaded" if inputs.der_subpanel_load_kw is not None else "⚠ Not loaded",
        width=200
    )
    main_load_status = pn.widgets.StaticText(
        name="Main Load Data",
        value="✓ Loaded" if inputs.main_panel_load_kw is not None else "⚠ Not loaded",
        width=200
    )

    # Action buttons
    compute_btn = pn.widgets.Button(name="⚡ Compute", button_type="primary", width=100)
    copy_btn = pn.widgets.Button(name="📋 Copy", button_type="default", width=100)
    remove_btn = pn.widgets.Button(name="🗑 Remove", button_type="danger", width=100)

    # Output display - using Tabulator for proper table display
    initial_df = outputs.render_summary_table() if outputs.summary_stats is not None else pd.DataFrame({"Message": ["Click Compute to run optimization"]})
    output_display = pn.widgets.Tabulator(
        initial_df,
        name="Results",
        height=300,
        disabled=True,
        show_index=True,
        sizing_mode='stretch_width'
    )

    # Status indicator with loading support
    status_indicator = pn.indicators.LoadingSpinner(value=False, width=20, height=20)
    status_text = pn.widgets.StaticText(
        name="Status",
        value="Ready" if outputs.optimization_status is None else outputs.optimization_status,
        width=180
    )
    status_row = pn.Row(status_indicator, status_text)

    # ===== Container Layout =====

    scenario_box = pn.Column(
        pn.pane.Markdown(f"## Scenario {scenario.id}"),
        scenario_name_input,
        status_row,
        pn.Row(compute_btn, copy_btn, remove_btn),

        study_params_header,
        pn.Row(start_date_input, study_years_input),
        pn.Row(optimize_checkbox),

        sizing_header,
        pn.Row(min_battery_input, max_battery_input),
        pn.Row(min_solar_input, max_solar_input),

        uploads_header,
        pn.Row(load_example_btn),
        pn.Column(
            solar_upload,
            solar_status,
            der_load_upload,
            der_load_status,
            main_load_upload,
            main_load_status,
        ),

        output_display,

        sizing_mode="fixed",
        styles={
            "border": "2px solid #ddd",
            "padding": "15px",
            "margin": "10px",
            "background": "#f9f9f9",
            "border-radius": "5px"
        },
        width=400
    )

    # ===== Helper Functions =====

    def collect_inputs() -> SalesEngineeringInputs:
        """Collect all input values from widgets."""
        return SalesEngineeringInputs(
            id=scenario.id,
            scenario_name=scenario_name_input.value,
            start_date=start_date_input.value,
            study_years=study_years_input.value,
            optimize_sizing=optimize_checkbox.value,
            unit_solar_timeseries_kw=inputs.unit_solar_timeseries_kw,
            der_subpanel_load_kw=inputs.der_subpanel_load_kw,
            main_panel_load_kw=inputs.main_panel_load_kw,
            min_battery_units=min_battery_input.value,
            max_battery_units=max_battery_input.value,
            min_solar_units=min_solar_input.value,
            max_solar_units=max_solar_input.value,
        )

    def update_consolidated_results():
        """Update the consolidated results display."""
        all_outputs = list(scenario_outputs_registry.values())
        df = show_consolidated_results(all_outputs)
        # Update the Tabulator widget with the new dataframe
        consolidated_results_display.value = df

    # ===== Callbacks =====

    def on_optimize_change(event):
        """Enable/disable top_n_input based on optimize_sizing checkbox."""
        max_battery_input.disabled = not optimize_checkbox.value
        max_solar_input.disabled = not optimize_checkbox.value

    def on_solar_upload(event):
        """Handle solar data upload."""
        if solar_upload.value is not None:
            try:
                df = load_dataframe_from_csv(
                    solar_upload.value,
                    ['solar', 'ac_power_kw', 'power']
                )
                inputs.unit_solar_timeseries_kw = df
                solar_status.value = f"✓ Loaded ({len(df)} rows)"
                pn.state.notifications.success("Solar data loaded successfully!")
            except Exception as e:
                solar_status.value = f"✗ Error: {str(e)[:30]}"
                pn.state.notifications.error(f"Failed to load solar data: {e}")

    def on_der_load_upload(event):
        """Handle DER load upload."""
        if der_load_upload.value is not None:
            try:
                df = load_dataframe_from_csv(
                    der_load_upload.value,
                    ['der_subpanel_load', 'load_kw', 'load', 'circuit_load']
                )
                inputs.der_subpanel_load_kw = df
                der_load_status.value = f"✓ Loaded ({len(df)} rows)"
                pn.state.notifications.success("DER load data loaded successfully!")
            except Exception as e:
                der_load_status.value = f"✗ Error: {str(e)[:30]}"
                pn.state.notifications.error(f"Failed to load DER load: {e}")

    def on_main_load_upload(event):
        """Handle main panel load upload."""
        if main_load_upload.value is not None:
            try:
                df = load_dataframe_from_csv(
                    main_load_upload.value,
                    ['main_panel_load', 'main_load_kw', 'load', 'non_circuit_load']
                )
                inputs.main_panel_load_kw = df
                main_load_status.value = f"✓ Loaded ({len(df)} rows)"
                pn.state.notifications.success("Main load data loaded successfully!")
            except Exception as e:
                main_load_status.value = f"✗ Error: {str(e)[:30]}"
                pn.state.notifications.error(f"Failed to load main load: {e}")

    def on_load_example(event):
        """Load example data into the scenario."""
        try:
            # Load example CSVs from local files
            REF_DIR = '/Users/eric/Documents/Bidness/energyoptimizer/reference_data'
            base_path = Path(REF_DIR)
            solar_df = pd.read_csv(base_path / "solar.csv", parse_dates=['timestamp'], index_col='timestamp')
            der_load_df = pd.read_csv(base_path / "loads.csv", parse_dates=[0], index_col=0)
            main_load_df = pd.read_csv(base_path / "loads.csv", parse_dates=[0], index_col=0)

            inputs.unit_solar_timeseries_kw = solar_df
            inputs.der_subpanel_load_kw = der_load_df
            inputs.main_panel_load_kw = main_load_df

            solar_status.value = f"✓ Loaded ({len(solar_df)} rows)"
            der_load_status.value = f"✓ Loaded ({len(der_load_df)} rows)"
            main_load_status.value = f"✓ Loaded ({len(main_load_df)} rows)"

            pn.state.notifications.success("Example data loaded successfully!")
        except Exception as e:
            pn.state.notifications.error(f"Failed to load example data: {e}")

    def on_compute(event):
        """Run the optimization computation."""
        try:
            status_text.value = "Computing..."
            status_indicator.value = True  # Show loading spinner
            compute_btn.loading = True

            # Collect current inputs
            current_inputs = collect_inputs()

            # Run computation
            results = current_inputs.compute()

            # Update current scenario with first result
            updated_output = results[0]
            output_display.value = updated_output.render_summary_table()
            scenario_outputs_registry[scenario.id] = updated_output
            status_text.value = updated_output.optimization_status or "Complete"
            
            # Update scenario name if it was auto-generated
            if updated_output.scenario_name and not scenario_name_input.value:
                scenario_name_input.value = updated_output.scenario_name

            # If multiple results, create new scenarios for results[1:]
            if len(results) > 1:
                max_idx = max(scenarios_registry.keys()) if scenarios_registry else 0
                for i, result_output in enumerate(results[1:], 1):
                    new_idx = max_idx + i
                    
                    # Set min/max units based on sizing results
                    if result_output.sizing_results:
                        n_batt = int(result_output.sizing_results.get('n_batt_blocks', 0))
                        n_solar = int(result_output.sizing_results.get('n_solar', 0))
                        new_inputs = attrs.evolve(
                            current_inputs,
                            id=new_idx,
                            scenario_name=result_output.scenario_name,
                            min_battery_units=n_batt,
                            max_battery_units=n_batt,
                            min_solar_units=n_solar,
                            max_solar_units=n_solar
                        )
                    else:
                        new_inputs = attrs.evolve(current_inputs, id=new_idx, scenario_name=result_output.scenario_name)
                    
                    # Create new scenario with same inputs but different output
                    new_scenario = SalesEngineeringScenario(
                        id=new_idx,
                        inputs=new_inputs,
                        outputs=attrs.evolve(result_output, id=new_idx)
                    )
                    add_callback(new_idx, new_scenario)

            # Update consolidated results
            update_consolidated_results()
            pn.state.notifications.success(f"Computation complete! Generated {len(results)} scenario(s).")

        except Exception as e:
            status_text.value = f"Error"
            output_display.value = pd.DataFrame({"Error": [str(e)]})
            pn.state.notifications.error(f"Computation failed: {e}")
        finally:
            compute_btn.loading = False
            status_indicator.value = False  # Hide loading spinner

    def on_copy(event):
        """Copy this scenario to a new one."""
        max_idx = max(scenarios_registry.keys()) if scenarios_registry else 0
        new_id = max_idx + 1

        # Create a copy with new ID
        copied_inputs = attrs.evolve(collect_inputs(), id=new_id, scenario_name=f"Copy of {scenario_name_input.value}")
        copied_scenario = SalesEngineeringScenario(
            id=new_id,
            inputs=copied_inputs,
            outputs=SalesEngineeringOutputs(id=new_id, scenario_name=copied_inputs.scenario_name)
        )

        add_callback(new_id, copied_scenario)
        pn.state.notifications.info(f"Scenario copied to ID {new_id}")

    def on_remove(event):
        """Remove this scenario."""
        if len(scenarios_registry) <= 1:
            pn.state.notifications.warning("Cannot remove the last scenario!")
            return

        scenarios_registry.pop(scenario_idx, None)
        scenario_outputs_registry.pop(scenario_idx, None)
        refresh_callback()
        pn.state.notifications.info(f"Scenario {scenario.id} removed")

    # ===== Wire up callbacks =====

    optimize_checkbox.param.watch(on_optimize_change, 'value')
    solar_upload.param.watch(on_solar_upload, 'value')
    der_load_upload.param.watch(on_der_load_upload, 'value')
    main_load_upload.param.watch(on_main_load_upload, 'value')
    load_example_btn.on_click(on_load_example)
    compute_btn.on_click(on_compute)
    copy_btn.on_click(on_copy)
    remove_btn.on_click(on_remove)

    return scenario_box


# ============================================================================
# Main Application
# ============================================================================

def create_app():
    """Create and return the main Panel application."""

    def refresh_scenarios():
        """Rebuild the container from current scenario dict."""
        scenarios_container.objects = list(scenarios.values())

    def add_scenario(scenario_idx: int, scenario: SalesEngineeringScenario):
        """Add a new scenario to the registry and refresh UI."""
        scenario_ui = create_scenario_ui(
            scenario_idx,
            scenario,
            scenarios,
            scenario_outputs,
            scenarios_container,
            consolidated_results_display,
            refresh_scenarios,
            add_scenario
        )
        scenarios[scenario_idx] = scenario_ui
        refresh_scenarios()

    def on_add_scenario(event):
        """Create a new scenario."""
        next_idx = max(scenarios.keys()) + 1 if scenarios else 1
        scenario_name = f"Scenario {next_idx}"
        default_scenario = SalesEngineeringScenario(
            id=next_idx,
            inputs=SalesEngineeringInputs(id=next_idx, scenario_name=scenario_name),
            outputs=SalesEngineeringOutputs(id=next_idx, scenario_name=scenario_name)
        )
        add_scenario(next_idx, default_scenario)
        pn.state.notifications.info(f"Scenario {next_idx} added")

    scenarios = {}
    scenario_outputs = {}

    # UI containers
    scenarios_container = pn.Row(
        styles={"overflow-x": "auto", "flex-wrap": "nowrap"},
        # sizing_mode="stretch_width"
    )

    consolidated_results_display = pn.widgets.Tabulator(
        pd.DataFrame({"Message": ["No results available"]}),
        name="Consolidated Results",
        height=300,
        disabled=True,
        show_index=False,
        sizing_mode='stretch_width'
    )

    # Add Scenario button
    add_btn = pn.widgets.Button(name="➕ Add Scenario", button_type="success", width=150)
    add_btn.on_click(on_add_scenario)

    # Initialize with one scenario
    initial_scenario = SalesEngineeringScenario(
        id=1,
        inputs=SalesEngineeringInputs(id=1, scenario_name="Scenario 1"),
        outputs=SalesEngineeringOutputs(id=1, scenario_name="Scenario 1")
    )
    add_scenario(1, initial_scenario)

    app = pn.Column(
        pn.pane.Markdown("# Sales Engineering: Solar + Battery Sizing"),
        pn.pane.Markdown("""
        This application helps size solar and battery systems for commercial installations.
        Upload your site data or use example data, configure parameters, and run optimizations.
        """),
        add_btn,
        scenarios_container,
        pn.pane.Markdown("## Consolidated Results"),
        consolidated_results_display,
        sizing_mode="stretch_width"
    )
    return app

app = create_app()
app.servable()


if __name__ == "__main__":
    pn.serve(app, show=True, port=5006)
