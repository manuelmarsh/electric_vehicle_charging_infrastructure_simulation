from modules.simulated_object import SimulatedObject


class Grid(SimulatedObject):
    """
    Class implementing the Grid physical model.

    Parameters
    ----------
    SimulatedObject : Class implenting a genering simulated object
    """

    def __init__(self,
                 pacmax: float = 0) -> None:
        self.pacmax = pacmax  # Max power available from the grid

        self.saturated = False  # Max grid power reached
        self.tot_preq = 0  # Power request
        self.pin = 0  # Power absorbed from the grid

    def setInput(self,
                 pacmax: float = 0,
                 tot_preq: float = 0) -> None:

        self.pacmax = pacmax
        self.tot_preq = tot_preq  # = sum(pin_mpu)

    def step(self) -> None:

        self.pin = min(self.tot_preq, self.pacmax)
        self.saturated = self.tot_preq >= self.pacmax

    def getOutput(self) -> float:

        return self.pin

    def reached_saturation(self) -> bool:

        return self.saturated
