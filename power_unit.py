from modules.simulated_object import SimulatedObject
import numpy as np


def mpu_p_iter_calc(pn: np.array,
                    xj: np.array,
                    preq_out: float = 0) -> np.array:
    """
    Calculates the power allocation for a given array
    of nominal power values and a total power request.

    Parameters
    ----------
    pn : np.array
        array of nominal powers of each power unit
    xj : np.array
        array of the active connections with pn array
    preq_out : float
        requested output power

    Returns
    -------
    np.array
        array of output powers for each power unit
    """

    n = sum(xj)  # number of connected mpus
    p = np.zeros(shape=len(pn))

    if n == 0:  # if no mpu connected to puj
        return p

    pi = preq_out/n  # initial power guess, equally divided for each mpu
    dp = 0

    for k in range(len(pn)):

        for i in range(len(pn)):

            if xj[i] != 0:  # mpu is connected

                # power absorbed by mpu[i] is lower than its nominal power
                # ...continue the iteration
                if p[i] < pn[i]:

                    # initial power is lower than the mpu[i] nominal power
                    # ...assign more power
                    if pi < pn[i]:

                        p[i] = pi + dp/n

                        # check if mpu nominal power is reached
                        # ...if so, mpu[i] absorbed power is found,
                        # calculate exceeding power and restart iteration
                        if p[i] > pn[i]:
                            dp += pi - pn[i]
                            p[i] = pn[i]
                            n -= 1
                            break

                    # initial power is higher than the mpu[i] nominal power
                    # ...mpu[i] absorbed power is found,
                    # calculate exceeding power and restart iteration
                    else:
                        dp += pi - pn[i]
                        p[i] = pn[i]
                        n -= 1
                        break
    return p


class PowerUnit(SimulatedObject):
    """
    Class implementing the Power Unit physical model.

    Parameters
    ----------
    SimulatedObject : Class implenting a genering simulated object
    """

    def __init__(self,
                 pn: float = 0,
                 eff: float = 1):
        self.pn = pn
        self.eff = eff

        self.saturated = False
        self.pin = 0
        self.preq_out = 0

    def setInput(self,
                 pn: float = 0,
                 eff: float = 1,
                 preq_out: float = 0):
        self.pn = pn
        self.eff = eff
        self.preq_out = preq_out

    def step(self) -> None:
        self.pin = min(self.preq_out, self.pn) / self.eff
        self.saturated = self.preq_out >= self.pn

    def getOutput(self) -> float:
        return self.pin

    def reached_saturation(self) -> bool:
        return self.saturated
