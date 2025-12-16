from modules.grid import Grid
from modules.power_unit import PowerUnit
from modules.plug_unit import PlugUnit
from modules.electric_vehicle import ElectricVehicle
from optimizer.optimizer import Optimizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_arrays_from_row(row: pd.Series,
                        n_row: int = 0,
                        n_col: int = 0) -> list:
    """
    Get arrays from a given row of the input file of the simulation.
    The only important constraint is the correct order of the paramaters.

    Parameters
    ----------
    row : pd.Series
        Row in the input file. The correct order of inputs is:
        Pacmax, Pn, eff, Preq, En, iSoc, Cev, x_avl
        Array-like inputs has to be given in an ascending order of
        positions indexes in DC matrix.
        x_avl is considered as a row_wise flat matrix.
    n_row : int, optional
        number of rows of the 'DC matrix', i.e. number of MPUs, by default 0
    n_col : int, optional
        number of columns of the DC matrix, i.e. number of PUs, by default 0

    Returns
    -------
    list
        List of returned data:
        - pacmax: grid max power
        - pn: array of MPUs nominal power
        - eff: array of MPUs efficiency
        - preq: array of EVs requested power
        - en: array of EVs nominal energy
        - isoc: array of EVs initial state of charge
        - cev: array of EVs connection status
        - x_avl: array of connection availability matrix

    Raises
    ------
    Exception
        When the number of columns in the input file is wrong
    """

    if len(row) != (1 + 2*n_row + 4*n_col + n_row*n_col):
        raise Exception('The number of columns in the input file is wrong.')
    pacmax = row.iloc[0]

    pn_serie = row.iloc[1:(1+n_row)]
    pn = np.array(pn_serie)

    eff_serie = row.iloc[(1+n_row):(1+2*n_row)]
    eff = np.array(eff_serie)

    preq_serie = row.iloc[(1+2*n_row):(1+2*n_row+n_col)]
    preq = np.array(preq_serie)

    en_serie = row.iloc[(1+2*n_row+n_col):(1+2*n_row+2*n_col)]
    en = np.array(en_serie)

    soc_serie = row.iloc[(1+2*n_row+2*n_col):(1+2*n_row+3*n_col)]
    soc = np.array(soc_serie)

    cev_serie = row.iloc[(1+2*n_row+3*n_col):(1+2*n_row+4*n_col)]
    cev = np.array(cev_serie)

    x_avl_serie = row.iloc[(1+2*n_row+4*n_col):]
    x_avl = np.reshape(np.array(x_avl_serie), newshape=(n_row, n_col))

    return pacmax, pn, eff, preq, en, soc, cev, x_avl


def at_least_one_mpu_not_used(x: np.array,
                              x_avl: np.array) -> bool:     # all busy == NOT (at least one not used)
    """
    Checks if at least one MPU is not used in the current iteration.

    Parameters
    ----------
    x : np.array
        Array of the current connections in the DC matrix.
    x_avl : np.array
        Array of the available connections in the DC matrix.

    Returns
    -------
    bool
        True if theere is at least one MPU not used, False otherwise
    """
    num_rows = x.shape[0]

    for i in range(num_rows):
        if sum(x[i]) == 0:          # MPU[i] not connected to any PU
            if sum(x_avl[i]) != 0:  # and at least 1 connection is available
                return True         # --> MPU[i] is NOT busy
    return False


def disconnect_full_evs(cev_np: np.array,
                        cev_prev_np: np.array,
                        ev_is_full: np.array) -> np.array:
    """
    Disconnect EV (set cev=0) if its battery is full.

    Parameters
    ----------
    cev_np : np.array
        Current iteration's connections array between PU-EV at position j.
    cev_prev_np : np.array
        Previous iteration's connections array between PU-EV at position j.
    ev_is_full : np.array
        Previous iteration's boolean array where 1 indicates the EV's
        battery is full. The array is changed in place.

    Returns
    -------
    np.array
        Updated connections
    """
    cev_fake_np = np.copy(cev_np)
    for j in range(len(cev_np)):
        if cev_prev_np[j] == cev_np[j]:  # If same EV is still connected
            if ev_is_full[j]:            # If EV's battery is full
                cev_fake_np[j] = 0            # --> disconnect
    return cev_fake_np


def estimate_pev_np(pacmax: float,
                    pabs_np: np.array,
                    pev_np_prev: np.array,
                    pev_found: np.array,
                    x_np: np.array,
                    x_avl_np: np.array,
                    cev_np: np.array,
                    pn_np: np.array,
                    ) -> np.array:
    """
    Calculates an estimation of the power requested by the EVs.
    The estimation algorhtm is based on increasing step by step the power
    that is available to a plug unit, starting from the nominal power of the
    smallest power unit, until the power requested by the EV is reached or
    until one component has reached its maximum available power.

    Parameters
    ----------
    pacmax : float
        Max power available from the grid.
    pabs_np : np.array
        Array of power absorbed by the EVs at the previous iteration.
    pev_np_prev : np.array
        Power request estimation at the previous iteration.
    pev_found : np.array
        Array of boolean telling if the power request estimation matches the
        real power request.
    x_np : np.array
        Connection matrix at the previous iteration.
    x_avl_np : np.array
        Connection availability matrix at the current iteration.
    cev_np : np.array
        Array of connections between PU-EV at the current iteration.
    pn_np : np.array
        Array of nominal powers at the current iteration.

    Returns
    -------
    np.array
        Power request estimation for each EV.
    """

    # The increment of power for the estimation algorithm is
    # equal to the nominal power of the smallest power unit
    dp = min(pn_np)
    pev_np = np.zeros(shape=len(pev_np_prev))

    for j in range(len(pev_np)):
        pabs = pabs_np[j]
        pev_prev = pev_np_prev[j]
        cev = cev_np[j]

        if cev == 0:
            pev = dp
            # in case it just turned 0 --> reset to False (0)
            pev_found[j] = 0

        else:  # cev == 1:
            pavl = sum(x_np[:, j]*pn_np)
            pabs = pabs_np[j]

            if pabs < pavl:
                pev = pabs
                pev_found[j] = 1  # True

            else:  # pabs == pavl:
                if sum(pabs_np) >= pacmax:  # Limited by grid insufficiency
                    pev = pev_prev
                # Limited by insufficient connection availability
                elif not at_least_one_mpu_not_used(x_np, x_avl_np):
                    pev = pev_prev
                # Limited by real requested power preq
                elif pev_found[j]:
                    pev = pev_prev
                else:  # Limited by wrong previous estimation pev_prev
                    pev = pev_prev + dp

        pev_np[j] = pev

    return pev_np


def has_changed(current_arrays,
                previous_arrays) -> bool:
    # Compare each current array with its previous version
    for curr, prev in zip(current_arrays, previous_arrays):
        if not np.array_equal(curr, prev):
            return True
    return False


def simulate(input_file_path: str,
             num_mpus: int,
             num_pus: int,
             dt: float = 1,
             keff: float = 1,
             kvar: float = 1,
             show_p_matrix: bool = False,
             opt_timeout: float = None,  # = dt by default
             show_opt_calcs: bool = False) -> list:
    """
    Simulate the Electric Vehicle Charging Infrastructure.
    It consists of alogithmic (EV charging power estimation, optimizer)
    and physical simulations (EV battery, power modules, etc.).
    The physical simulation models execution order it the reverse of the real
    power flow, i.e. from the EV to the Power Units.

    Each simulation physical component share the same structure,
    with 2 inputs and one output:
    - Input: required output power (ex. MPU required output power)
    - Input available power
    - Output: calculated input power (ex. MPU calculated input power
              considering the module efficiency)

    The physical simulation part consists of a "fast" loop cycling all the
    physical models until one component reaches a saturation condition (for ex.
    EV requested power is reached).

    Parameters
    ----------
    input_file_path : str
        Path to the file containing the inputs for the simulation
    num_mpus : int
        Number of MPU (Modular Power Units) in the EVCI
    num_pus : int
        Number of PU (Plug Units) in the EVCI
    dt : float, optional
        Simulation step size in seconds, by default 1
    keff : float, optional
        Efficiency gain for the optimization problem objective function,
        by default 1
    kvar : float, optional
        Gain to tune the number of contactors matrix variation for the
        optimization problem objective function, by default 1
    show_p_matrix : bool, optional
        Set True to enable the plot of contactors matrix heatmap,
        by default False
    opt_timeout : float, optional
        Optimizer timeout, by default None

    Returns
    -------
    list
        List of returned data:
        - grid: Grid object
        - mpu_dict: PowerUnit objects' dictionary
        - pu_dict: PlugUnit objects' dictionary
        - ev_dict: ElectricVehicle objects' dictionary
        - time_np: Time instants array
        - pabs_dict: dictionary like {j_position: np.array} i.e. evolution over
            time of Absorbed Power for every EV
        - pev_dict: evolutions over time of EV charge power estimation for
            each EV
        - preq_dict: evolutions over time of EV charge power request for
            each EV
        - soc_dict: evolutions over time of EV state of charge for each EV
        - pfrom_mpus_dict: dictionary like {j_position: np.array} i.e.
            evolution over time of power delivered by each MPU to each PU
        - pn_dict: dictionary like {i_position: float} i.e. nominal power of
            each MPU
        - p_tot_t: evolution over time of total power delivered
        - p_notused_t: evolution over time of total unused power
        - n_var_t: evolution over time of number of connections' variations

    """

    # Create objects
    grid = Grid()

    # We chose this style of keys: 'i1','i2'... for i positions (mpus),
    # same with j for j positions (pus, evs)
    mpu_dict = {f'i{i+1}': PowerUnit() for i in range(num_mpus)}
    pu_dict = {f'j{j+1}': PlugUnit() for j in range(num_pus)}
    ev_dict = {f'j{j+1}': ElectricVehicle() for j in range(num_pus)}

    i_keys = list(mpu_dict.keys())
    j_keys = list(pu_dict.keys())

    df = pd.read_excel(input_file_path, header=0, index_col=0)
    num_steps = df.shape[0]
    tot_time = df.shape[0]*dt
    time_np = np.arange(0, tot_time, dt)  # list with time iterations from 0

    # Plotting variables to memorize evolution
    p_tot_t = np.zeros(shape=len(time_np))
    p_notused_t = np.zeros(shape=len(time_np))
    n_var_t = np.zeros(shape=len(time_np))

    # dict {'j0': np.array([pabs_time0, pabs_time1, ..])}
    pabs_dict = {key: np.zeros(shape=num_steps) for key in j_keys}
    pev_dict = {key: np.zeros(shape=num_steps) for key in j_keys}
    preq_dict = {key: np.zeros(shape=num_steps) for key in j_keys}
    soc_dict = {key: np.zeros(shape=num_steps) for key in j_keys}
    pfrom_mpus_dict = {key: np.zeros(
        shape=[num_steps, num_mpus]) for key in j_keys}

    # Initialize variables for simulation

    # Initialize variables for simulation
    x_prev_np = np.zeros(shape=(num_mpus, num_pus))
    cev_prev_np = np.zeros(shape=num_pus)
    x_np = np.zeros(shape=(num_mpus, num_pus))
    pev_np = np.zeros(shape=num_pus)
    pabs_np = np.zeros(shape=num_pus)
    pev_found = np.zeros(shape=num_pus)
    ev_is_full = np.zeros(shape=num_pus)

    # Assign opt_time_limit = dt (as it is in real contest) if no other values
    # are provided
    if opt_timeout is None:
        opt_timeout = dt

    # Optimizer constructor
    optimizer = Optimizer(keff,
                          kvar,
                          timeout=opt_timeout,
                          show_calcs=show_opt_calcs)

    prev_inputs = []

    # Main simulation loop ----------------------------------------------------
    # 1. Pev estimation
    # 2. Optimizer execution
    # 3. Physical models simulation
    for iteration in range(0, len(time_np)):

        # the row of input file with data of this iteration
        row = df.loc[iteration]
        dp = 0.1  # [kW] power increment for the physical models simulation

        (pacmax, pn_np, eff_np, preq_np, en_np,
         isoc_np, cev_np, x_avl_np) = get_arrays_from_row(row,
                                                          num_mpus,
                                                          num_pus)

        # EV is "disconnected" when its SoC = 100%, simulating the end of
        # charge. Cev is forced to 0 so that Pev estimation and then optimizer
        # de-allocate MPUs connected to that PU.
        cev_fake_np = disconnect_full_evs(cev_np,
                                          cev_prev_np,
                                          ev_is_full)

        # Pev estimation
        pev_np = estimate_pev_np(pacmax,
                                 pabs_np,
                                 pev_np,
                                 pev_found,
                                 x_np,
                                 x_avl_np,
                                 cev_fake_np,
                                 pn_np)

        inputs = [pn_np, cev_fake_np, x_avl_np, pev_np, x_prev_np]

        if has_changed(inputs, prev_inputs) or iteration == 0:
            # Optimization model execution
            x_np, n_var, optimal_solution = optimizer.execute_opt_model(
                i_keys,
                j_keys,
                pacmax,
                pn_np,
                cev_fake_np,
                x_avl_np,
                pev_np,
                x_prev_np)

            prev_inputs = inputs
        else:
            print("No changes in inputs, skipping optimizer execution")

        pdel_np = np.zeros(shape=num_pus)

        # Physical models simulation
        for p in range(1, int((pacmax+1)/dp)):

            # Cycle through all PUs
            for j in range(num_pus):

                key = j_keys[j]
                pu = pu_dict[key]
                ev = ev_dict[key]
                # ... EV connected (Cev == 1)
                if cev_np[j] == 1:
                    # PU not saturated (available power not reached)
                    if not pu.reached_saturation():
                        # EV not saturated (requested power not reached)
                        if not ev.reached_saturation():
                            # EV's battery not full (SoC 1 not reached)
                            if not ev.battery_full():
                                pdel_np[j] = p*dp

            # EV
            pabs_np = np.zeros(shape=num_pus)  # vector j
            for j in range(num_pus):
                ev = ev_dict.get(j_keys[j])

                ev.setInput(preq_np[j],
                            pdel_np[j],
                            en_np[j],
                            isoc_np[j],
                            cev_np[j])
                ev.step()
                pabs_np[j] = ev.getOutput()

            # PU
            pout_mpus_np = np.zeros(shape=(num_mpus, num_pus))   # matrix ij
            for j in range(num_pus):
                pu = pu_dict.get(j_keys[j])
                preq_out_pu = pabs_np[j]

                pu.setInput(x_np, pn_np, j, preq_out_pu)
                pu.step()
                pout_mpus_np[:, j] = pu.getOutput()

            # MPU
            pin_mpus_np = np.zeros(shape=num_mpus)   # vector i
            for i in range(num_mpus):
                mpu = mpu_dict.get(i_keys[i])
                pn = pn_np[i]
                eff = eff_np[i]
                preq_out_mpu = sum(pout_mpus_np[i])

                mpu.setInput(pn, eff, preq_out_mpu)
                mpu.step()
                pin_mpus_np[i] = mpu.getOutput()

            # Grid
            tot_preq = sum(pin_mpus_np)
            grid.setInput(pacmax,
                          tot_preq)
            grid.step()
            # pin_grid = grid.getOutput()

            if any((all(ElectricVehicle.reached_saturation(value)
                        for value in ev_dict.values()),
                    all(ElectricVehicle.battery_full(value)
                        for value in ev_dict.values()),
                    all(PlugUnit.reached_saturation(value)
                        for value in pu_dict.values()),
                    all(PowerUnit.reached_saturation(value)
                        for value in mpu_dict.values()),
                    grid.reached_saturation())):
                break

        # Set new SOC and update ev_is_full array
        for j in range(num_pus):
            ev = ev_dict.get(j_keys[j])
            ev.setSoc(dt)
            if ev.battery_full():
                ev_is_full[j] = 1
            else:
                ev_is_full[j] = 0

        # Update memory for outputs
        if show_p_matrix:
            # Show pout_mpus
            mat = pd.DataFrame(pout_mpus_np, index=i_keys, columns=j_keys)
            mat_filtered = mat[mat != 0].fillna(0).astype(int)
            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(mat_filtered, annot=True, fmt='d',
                        cmap='Blues', cbar=False,
                        linewidths=0.5, linecolor='grey',
                        vmin=0, vmax=min(pn_np))
            plt.title(f'''Power delivered from MPU[i] to PU[j]
                          at time iteration {iteration}''')
            plt.xlabel('Plug Units')
            plt.ylabel('Power Units')
            plt.show()

        # Update plotting variables of evolutions in time
        pavl_np = np.zeros(shape=num_pus)

        for j in range(num_pus):
            pavl_np[j] = sum(x_np[:, j]*pn_np)

            pabs_t = pabs_dict.get(j_keys[j])
            pabs_t[iteration] = pabs_np[j]

            pev_t = pev_dict.get(j_keys[j])
            pev_t[iteration] = pev_np[j]

            preq_t = preq_dict.get(j_keys[j])
            preq_t[iteration] = preq_np[j]

            soc_t = soc_dict.get(j_keys[j])
            soc_t[iteration] = ev_dict[j_keys[j]].getSoc()

            pfrom_mpus_t = pfrom_mpus_dict.get(j_keys[j])
            # np.array([pfrom_mpu0, pfrom_mpu1, ..., pfrom_mpui, ...]) ==
            # column j of pout_mpus_np
            pfrom_mpus_t[iteration] = pout_mpus_np[:, j]

        p_tot_t[iteration] = sum(pabs_np)
        p_notused_t[iteration] = sum(pavl_np) - sum(pabs_np)
        n_var_t[iteration] = n_var

        if not optimal_solution:
            if show_opt_calcs:
                print("Optimizer didn't find the optimal solution")
        else:
            # current DC matrix status is updated only if solver found an
            # optimal solution
            x_prev_np = np.copy(x_np)

        cev_prev_np = np.copy(cev_np)

    # Pns are considered constant over time,
    # if they are not we are using Pns at last iteration
    pn_dict = {i_keys[i]: pn_np[i] for i in range(num_mpus)}

    return (grid, mpu_dict, pu_dict, ev_dict, time_np, pabs_dict, pev_dict,
            preq_dict, soc_dict, pfrom_mpus_dict, pn_dict,
            p_tot_t, p_notused_t, n_var_t)


def create_random_colors_dict(keys: list):
    colormap = plt.cm.get_cmap('tab20', len(keys))
    colors = {keys[i]: colormap(i / len(keys)) for i in range(len(keys))}
    return colors


def plot_sim_results(time_np,
                     pabs_dict,
                     pev_dict,
                     preq_dict,
                     soc_dict,
                     pfrom_mpus_dict,
                     pn_dict,
                     p_tot_t,
                     p_notused_t,
                     n_var_t,
                     time_window=3600):

    i_keys = list(pn_dict.keys())
    j_keys = list(pabs_dict.keys())
    num_mpus = len(i_keys)
    num_pus = len(j_keys)
    i_colors = create_random_colors_dict(i_keys)
    j_colors = create_random_colors_dict(j_keys)

    # Print average KPIs per each time window: --------------------------------
    # 1. Total power delivered
    # 2. Total power not used
    # 3. Number of variations of connections
    if time_window >= time_np[-1]:
        print(f'''During first {time_np[-1]} seconds: \n
    {'Average total delivered power:'.ljust(50)} {np.average(p_tot_t):.2f} kW
    {'Average unused power:'.ljust(50)} {np.average(p_notused_t):.2f} kW
    {'Average number of variations of connections:'.ljust(50)} {np.average(n_var_t):.2f}\n
        ''')
    else:
        counter = 1
        i_start = 0
        i_end = np.where(time_np > time_window)[0][0]
        i_max = len(time_np)-1
        t_end = time_window

        while i_start <= i_max:

            print(f'''\n During slot {counter} of {time_window} seconds: \n\n
        {'Average total delivered power:'.ljust(50)}
            {np.average(p_tot_t[i_start : i_end]):.2f} kW
        {'Average unused power:'.ljust(50)}
            {np.average(p_notused_t[i_start : i_end]):.2f} kW
        {'Average number of variations of connections:'.ljust(50)}
            {np.average(n_var_t[i_start :i_end]):.2f}\n''')

            counter += 1
            i_start = i_end
            t_end += time_window
            if t_end < time_np[-1]:
                i_end = np.where(time_np > t_end)[0][0]  # first index
            else:
                i_end = i_max + 1

    # Ptot, PnotUsed ----------------------------------------------------------
    plt.figure(figsize=(15, 5))
    plt.scatter(time_np, p_tot_t, label='Power delivered')
    plt.scatter(time_np, p_notused_t, label='Power not used')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [kW]')
    plt.legend()
    plt.show()

    # Nvar over time ----------------------------------------------------------
    plt.figure(figsize=(15, 5))
    plt.scatter(time_np, n_var_t)
    plt.xlabel('Time [s]')
    plt.ylabel('Number of variations')
    plt.show()

    # Plot Pabs, Preq ---------------------------------------------------------
    plt.figure(figsize=(15, 5))

    for j in range(num_pus):

        key = j_keys[j]
        pabs_t = pabs_dict.get(key)
        # pev_t = pev_dict.get(key)
        preq_t = preq_dict.get(key)

        plt.scatter(time_np, pabs_t,
                    label=f'Pabs position {key}', color=j_colors[key])
        plt.step(time_np, preq_t,
                 label=f'Preq position {key}',
                 color=j_colors[key], linestyle='--')
        # plt.scatter(time_np, pev_t, label=f'Pev position {key}',
        #             color=j_colors[key], linestyle=':', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Power [kW]')
    plt.legend()
    plt.show()

    # Plot SoC ----------------------------------------------------------------
    plt.figure(figsize=(15, 5))

    for j in range(num_pus):
        key = j_keys[j]
        soc_t = soc_dict.get(key)
        plt.scatter(time_np, soc_t*100,
                    label=f'SoC position {key}', color=j_colors[key])
    plt.step(time_np, np.ones(len(time_np))*100, linestyle='--', color='black')

    plt.xlabel('Time [s]')
    plt.ylabel('SoC [%]')
    plt.legend()
    plt.show()

    # Bar graph of Pn of each MPU (considered constant ------------------------
    # over all time iterations)
    plt.figure(figsize=(15, 5))

    for i, (key, pn) in enumerate(pn_dict.items()):
        plt.bar(i, pn, color=i_colors[key], width=0.3, label=f'Pn {key}')
        plt.title('Nominal power pn of each MPU')
        plt.legend()

    plt.xticks(ticks=list(range(num_mpus)), labels=i_keys)
    plt.show()

    # Bra graph with fractions of power delivered to the EVs by each MPU ------
    max_abs = max(np.concatenate(list(pabs_dict.values())))
    delta_t = np.min(np.diff(time_np))  # passo temporale minimo
    bar_width = delta_t * 0.8 
    
    for j in range(num_pus):
        plt.figure(figsize=(15, 5))
        # common upper ylim == max power absorbed over all times
        pfrom_mpus_t = pfrom_mpus_dict.get(j_keys[j])
        plt.ylim(-1, max_abs)

        # Create bars over time (x axis is time)
        for t in range(len(time_np)):
            bottom = 0  # Start from bar's bottom (height 0)
            for i, frac in enumerate(pfrom_mpus_t[t]):
                plt.bar(time_np[t], frac, bottom=bottom,
                        color=i_colors[i_keys[i]], width=bar_width)
                bottom += frac  # Increment bottom for next power fraction

        plt.xlabel('Time (s)')
        plt.ylabel('Power (kW)')
        plt.title(f'''Fractions of power absorbed by vehicle {j_keys[j]}
                  from each MPU over time''')
        plt.show()


if __name__ == '__main__':

    # Objective function gains
    keff = 1  # weights power not used
    kvar = 1  # weights number of contactors state variations

    # Simulation time step
    dt = 10  # seconds

    # EVCI infrastructure matrix size
    num_mpus = 4  # number of Modular Power Units
    num_pus = 3  # number of Plug Units

    # Time window for KPIs averaging

    input_file_path = (
        r"evci_input_4x3.xlsx")

    (grid, mpu_dict, pu_dict, ev_dict, time_np, pabs_dict, pev_dict,
     preq_dict, soc_dict, pfrom_mpus_dict, pn_dict, p_tot_t, p_notused_t,
     n_var_t) = simulate(keff=keff,
                         kvar=kvar,
                         show_p_matrix=False,
                         input_file_path=input_file_path,
                         dt=dt,
                         num_mpus=num_mpus,
                         num_pus=num_pus,
                         opt_timeout=dt,
                         show_opt_calcs=False)

    plot_sim_results(time_np,
                     pabs_dict,
                     pev_dict,
                     preq_dict,
                     soc_dict,
                     pfrom_mpus_dict,
                     pn_dict,
                     p_tot_t,
                     p_notused_t,
                     n_var_t,
                     time_window=20*60)
