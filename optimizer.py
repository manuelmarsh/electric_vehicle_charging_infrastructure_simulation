from pyomo.environ import (ConcreteModel, Set, Param, Var, Binary,
                           NonNegativeReals, Constraint, Objective,
                           SolverFactory, maximize, TerminationCondition)
import numpy as np


class Optimizer():

    def __init__(self,
                 keff: float,
                 kvar: float,
                 timeout: float,
                 show_calcs: bool) -> None:

        self.keff = keff
        self.kvar = kvar
        self.timeout = timeout
        self.show_calcs = show_calcs

    def execute_opt_model(self,
                          i_keys: dict,
                          j_keys: dict,
                          pacmax: float,
                          pn: np.array,
                          cev: np.array,
                          x_avl: np.array,
                          pev: np.array,
                          x_prev: np.array):

        # Convert arrays into dicts
        rows = range(len(i_keys))       # rows' indexes iterator
        columns = range(len(j_keys))    # columns' indexes iterator

        Keff = self.keff
        Kvar = self.kvar
        Pn_dict = {i_keys[i]: pn[i] for i in rows}
        Cev_dict = {j_keys[j]: cev[j] for j in columns}
        x_avl_dict = {(i_keys[i], j_keys[j]): x_avl[i, j]
                      for i in rows for j in columns}
        Pev_dict = {j_keys[j]: pev[j] for j in columns}
        x_prev_dict = {(i_keys[i], j_keys[j]): x_prev[i, j]
                       for i in rows for j in columns}

        model = ConcreteModel()
        # Sets
        model.i = Set(initialize=Pn_dict.keys())
        model.j = Set(initialize=Pev_dict.keys())

        # Parameters
        model.Pn = Param(model.i, initialize=Pn_dict)
        model.Cev = Param(model.j, initialize=Cev_dict)
        model.Pev = Param(model.j, initialize=Pev_dict)
        model.Pacmax = Param(initialize=pacmax)

        # Penalties
        model.Keff = Param(initialize=Keff)
        model.Kvar = Param(initialize=Kvar)

        model.x_avl = Param(
            model.i, model.j, initialize=x_avl_dict, mutable=True)
        model.x_prev = Param(
            model.i, model.j, initialize=x_prev_dict, mutable=True)

        # Variables
        # Binary variable for the connection between i and j
        model.x = Var(model.i, model.j, domain=Binary)
        # Auxiliary variable for absolute difference
        model.delta = Var(model.i, model.j, domain=NonNegativeReals)
        # Power delivered to each plug unit j
        model.p = Var(model.j, domain=NonNegativeReals)

        # Objective functions, maximize the power delivered
        model.Ptot = Var(domain=NonNegativeReals)  # Total power delivered
        model.NotUsedP = Var(domain=NonNegativeReals)
        # Number of variations in connections respect to previous iteration
        model.nVar = Var(domain=NonNegativeReals)

        # Each power unit i can be connected to only one plug unit
        def singleConnection_rule(model, i):
            return sum(model.x[i, j] for j in model.j) <= 1
        model.singleConnection = Constraint(
            model.i, rule=singleConnection_rule)

        # Each plug unit j must have at least one power unit connected
        def allConnected_rule(model, j):
            return sum(model.x[i, j] for i in model.i) >= model.Cev[j]
        model.allConnected = Constraint(model.j, rule=allConnected_rule)

        # The power absorbed by each plug unit j is limited by the power of the
        # connected power units
        def powerAvailable_rule(model, j):
            return (sum(model.Pn[i] * model.x[i, j]
                        for i in model.i) >= model.p[j])
        model.powerAvailable = Constraint(model.j, rule=powerAvailable_rule)

        # The power absorbed by each plug unit j is limited by the power
        # requested by the EV
        def poweRequested_rule(model, j):
            # Works better in this way, although sometimes fails
            return model.p[j] <= model.Pev[j]*model.Cev[j]
        model.poweRequested = Constraint(model.j, rule=poweRequested_rule)

        # The total power absorbed is limited by the maximum available power
        def maxPower_rule(model):
            return sum(model.p[j] for j in model.j) <= model.Pacmax
        model.maxPower = Constraint(rule=maxPower_rule)

        # Connection availability
        def connAvailable_rule(model, i, j):
            return model.x[i, j] <= model.x_avl[i, j]
        model.connAvailable = Constraint(
            model.i, model.j, rule=connAvailable_rule)

        def delta_upper_bound_rule(model, i, j):
            return model.delta[i, j] >= model.x[i, j] - model.x_prev[i, j]
        model.delta_upper_bound = Constraint(
            model.i, model.j, rule=delta_upper_bound_rule)

        def delta_lower_bound_rule(model, i, j):
            return model.delta[i, j] >= model.x_prev[i, j] - model.x[i, j]
        model.delta_lower_bound = Constraint(
            model.i, model.j, rule=delta_lower_bound_rule)

        # Defines the objective function for total power delivered
        def TotalPower_rule(model):
            return model.Ptot == sum(model.p[j] for j in model.j)
        model.TotalPower = Constraint(rule=TotalPower_rule)

        # Unused power that is allocated to the plug
        def NotUsedPower_rule(model):
            return model.NotUsedP == sum((sum(model.Pn[i] * model.x[i, j]
                                              for i in model.i) - model.p[j])
                                         for j in model.j)
        model.NotUsedPower = Constraint(rule=NotUsedPower_rule)

        # Variation in connections
        def Variations_rule(model):
            return model.nVar == sum(model.delta[i, j]
                                     for i in model.i
                                     for j in model.j)
        model.Variations = Constraint(rule=Variations_rule)

        # Defines the objective function considering efficiency
        # We penalize the Not Used Power because the MPUs' efficiency
        # is maximum when they work near their Nominal Power
        def obj_rule(model):
            return model.Ptot - model.Keff * model.NotUsedP - (
                model.Kvar * model.nVar)

        # Objective function to maximize the total power delivered
        # considering efficiency
        model.obj = Objective(rule=obj_rule, sense=maximize)

        # Solve the model
        solver = SolverFactory('glpk')

        # add argument tee=True to show calculations
        opt_result = solver.solve(
            model, timelimit=self.timeout, tee=self.show_calcs)

        # Convert pyomo variable xij into numpy array
        n_rows = len(model.i)
        n_cols = len(model.j)
        x = np.zeros(shape=(n_rows, n_cols))
        for i in range(n_rows):
            i_name = i_keys[i]
            for j in range(n_cols):
                j_name = j_keys[j]
                x[i, j] = model.x[(i_name, j_name)].value

        n_var = model.nVar.value

        solver_tc = opt_result.solver.termination_condition

        if solver_tc in {TerminationCondition.maxTimeLimit,
                         TerminationCondition.feasible}:
            optimal_solution = False
        else:
            optimal_solution = True

        return x, n_var, optimal_solution
