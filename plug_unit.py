import numpy as np
from modules.simulated_object import SimulatedObject
from modules.power_unit import mpu_p_iter_calc


class PlugUnit(SimulatedObject):
    """
    Class implementing the Plug Unit physical model.

    Parameters
    ----------
    SimulatedObject : Class implenting a genering simulated object
    """

    def __init__(self,
                 pavl: float = 0) -> None:

        self.pavl = pavl  # Available power at the plug unit

        self.saturated = False  # Available power at the plug unit reached
        self.j = 0  # Index of plug
        self.x = np.zeros(1)  # Connection matrix for plug unit j
        self.pn = np.zeros(1)  # Pn of the power units connected to the plug j
        self.preq_out = 0  # Output power requested from plug unit j
        # Output power of power units connected to plug unit j
        self.pout_mpu = np.zeros(1)

    def setInput(self,
                 x: np.array,
                 pn: np.array,
                 j_index: int = 0,
                 preq_out: float = 0) -> None:

        self.j = j_index
        self.x = x
        self.pn = pn
        self.preq_out = preq_out

    def step(self) -> None:
        j = self.j
        xj = self.x[:, j]  # desired column
        pn = self.pn
        self.pavl = sum(xj*pn)
        self.pout_mpu = mpu_p_iter_calc(pn, xj, self.preq_out)
        self.saturated = self.preq_out >= self.pavl

    def getOutput(self) -> np.array:

        return self.pout_mpu

    def reached_saturation(self) -> bool:

        return self.saturated
