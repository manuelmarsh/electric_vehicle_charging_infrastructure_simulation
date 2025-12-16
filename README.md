# electric_vehicle_charging_infrastructure_simulation
Simulation environment to validate Electric Vehicle Charging Optimization Algorithms, focusing on DC Fast Charging Infrastructure.

# Project Requirements and Execution

## Prerequisites

### 1. Install Pyomo Library
Install the pyomo library in your IDE by running the following command in the console:

    pip install pyomo

### 2. Download and Set Up GLPK Solver
Download the GLPK solver from an official online source.
Place the GLPK executable inside the project folder.

## How to Run
Execute the main simulation script using the following command:

    python evci_simulation.py

## Note on Simulation Parameters
To modify the simulation parameters:
- Create a new Excel file containing the updated input data.
- Update the input_file_path variable in the main script so it points to the new Excel file.
