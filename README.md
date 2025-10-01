# energyoptimizer

Goals of this project:
- Extensible site optimization and control of flexibility resources, starting with solar+storage 
- Ultimate goal: allow for deferrable loads (appliances), thermostatic loads (heat pump water heaters, thermostats), interruptible loads (EV charging)
- Combined operational and financial modeling to facilitate

## Applications
This is designed so that **the same optimizer can be used in sales, design, online/on-device operation, and cloud-based control**. 
- Design: Scenario analysis for initial product design/analysis.  Targeting product design teams who are trying to understand what options and costs can make sense for customers.  Tied to webapp (to be deployed with stub demo at [emunsing/webscenarios](https://github.com/emunsing/webscenarios))
- Sales: Top-of-pipe web simple web application for system financial/operational modeling
- Operation: Feed live data into optimizer for on-hardware optimization
- Cloud-based control: Fleet-level aggregation into a VPP


## Architecture:
- Optimizer (optimizers.py): Core optimizer.
  - Currently implemented:
    - Simple self-consumption
    - TOU-based tariff optimization
    - TOU-based optimization with endogenous sizing optimization of battery and solar sizes
  - Roadmap:
    - Demand charge management

- Optimizer runner (optimization_runner.py):
  - Wraps the optimizer function and manages time indexing for working through scenarios

- FinancialInputs:
  - Proper cash flow modeling for computing annualized product cost
