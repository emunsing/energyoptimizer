import panel as pn
import attrs

pn.extension(sizing_mode="stretch_width")

@attrs.define
class WebInputs:
    id: int | None = None
    x: float = 1.0

@attrs.define
class WebOutputs:
    id: int | None = None
    x_squared: float | None = None

@attrs.define
class WebScenario:
    id: int
    inputs: WebInputs = attrs.field(default=WebInputs(id=id))
    outputs: WebOutputs = attrs.field(default=WebOutputs(id=id))

    def update_id(self, new_id: int) -> 'WebScenario':
        updated_inputs = attrs.evolve(self.inputs, id=new_id)
        updated_outputs = attrs.evolve(self.outputs, id=new_id)
        return attrs.evolve(self, id=new_id, inputs=updated_inputs, outputs=updated_outputs)

def compute_one_scenario(scenario_inputs: WebInputs) -> WebOutputs:
    x = scenario_inputs.x
    x_squared = x ** 2
    return WebOutputs(x_squared=x_squared, id=scenario_inputs.id)

def my_function(scenario_inputs: WebInputs) -> list[WebScenario]:
    outputs = compute_one_scenario(scenario_inputs)
    return [WebScenario(inputs=scenario_inputs, outputs=outputs, id=scenario_inputs.id)]

# --- Scenario UI factory ---
def create_scenario(scenario_idx: int, scenario: WebScenario):
    print(f"Creating scenario {scenario_idx}")
    inputs = scenario.inputs
    outputs = scenario.outputs

    print("Expecting scenario id to be ", scenario.id)
    x_input = pn.widgets.FloatInput(
        name=f"Scenario {scenario.id} (x)", value=inputs.x, width=100
    )
    compute_btn = pn.widgets.Button(name="Compute", button_type="primary", width=70)
    copy_btn = pn.widgets.Button(name="Copy", button_type="primary", width=70)
    remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=70)
    
    # Output field to display computation results
    output_x_squared = pn.widgets.StaticText(
        name=f"Output (x²)", 
        value=str(outputs.x_squared) if outputs.x_squared is not None else "Not computed",
        width=120
    )

    # container for this scenario
    box = pn.Column(
        pn.Row(x_input, compute_btn, copy_btn, remove_btn),
        output_x_squared,
        sizing_mode="fixed",
        styles={"border": "2px solid #888", "padding": "10px", "margin": "10px"},
    )

    def collect_inputs() -> WebInputs:
        return WebInputs(x=x_input.value, id=scenario.id)

    def collect_outputs() -> WebOutputs:
        computed_value = None
        if output_x_squared.value != "Not computed" and not output_x_squared.value.startswith("Error"):
            try:
                computed_value = float(output_x_squared.value)
            except ValueError:
                computed_value = None
        return WebOutputs(x_squared=computed_value, id=scenario.id)

    def to_web_scenario():
        scenario_inputs = collect_inputs()
        scenario_outputs = collect_outputs()
        return WebScenario(inputs=scenario_inputs, outputs=scenario_outputs, id=scenario.id)

    # actions
    def do_compute(event):
        """Compute x**2 and display the result"""
        try:
            scenario_inputs = collect_inputs()
            result = my_function(scenario_inputs)
            updated_self = result[0]
            result = updated_self.outputs
            output_x_squared.value = str(result.x_squared)
        except Exception as e:
            output_x_squared.value = f"Error: {str(e)}"

    def do_copy(event):
        print("Copying; current scenario ids: ", scenarios.keys())
        max_idx = max(scenarios.keys()) if scenarios else 0
        # Copy both the input value and the computed output
        current_scenario_snapshot = to_web_scenario()
        copied_scenario = current_scenario_snapshot.update_id(max_idx + 1)
        add_scenario(max_idx + 1, copied_scenario)

    def do_remove(event):
        print(f"Removing scenario {scenario_idx}")
        scenarios.pop(scenario_idx)
        refresh_scenarios()

    compute_btn.on_click(do_compute)
    copy_btn.on_click(do_copy)
    remove_btn.on_click(do_remove)

    return box

# --- Scenario registry and container ---
scenarios = {}
scenarios_container = pn.Row(styles={"overflow-x": "auto", "flex-wrap": "nowrap"})

def refresh_scenarios():
    """Rebuild the container from current scenario dict."""
    scenarios_container.objects = list(scenarios.values())


def add_scenario(scenario_idx: int, scenario: WebScenario):
    print(f"Adding scenario {scenario_idx}")
    scenarios[scenario_idx] = create_scenario(scenario_idx, scenario)
    refresh_scenarios()

# --- "Add Scenario" button ---
add_btn = pn.widgets.Button(name="Add Scenario", button_type="success")

def do_add(event):
    next_idx = max(scenarios.keys()) + 1 if scenarios else 1
    default_scenario = WebScenario(id=next_idx)
    print(f"Adding scenario with id {next_idx}")
    add_scenario(scenario_idx=next_idx, scenario=default_scenario)

add_btn.on_click(do_add)

# --- App Layout ---
app = pn.Column(
    "# MVP: Dynamic Scenarios with x²",
    add_btn,
    scenarios_container,
)

app.servable()

# Launch
if __name__ == "__main__":
    pn.serve(app, show=True)