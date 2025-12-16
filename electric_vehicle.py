from modules.simulated_object import SimulatedObject


class ElectricVehicle(SimulatedObject):
    """
    Class implementing the Electric Vehicle physical model.

    Parameters
    ----------
    SimulatedObject : Class implenting a genering simulated object
    """

    def __init__(self,
                 preq: float = 0,
                 en: float = 0,
                 isoc: float = 0) -> None:

        self.preq = preq  # Charge power request
        self.en = en  # Nominal energy
        self.isoc = isoc  # Initial SoC
        self.soc = isoc  # SoC

        self.saturated = False  # Charge power request reached
        self.cev_prev = 0  # Previous EV connection status
        self.pdel = 0  # Power delivered to the EV
        self.cev = 0  # EV connection status
        self.pabs = 0   # Power absorbed by the EV

    def setInput(self,
                 preq: float = 0,
                 pdel: float = 0,
                 en: float = 0,
                 isoc: float = 0,
                 cev: int = 0) -> None:

        self.preq = preq
        self.pdel = pdel
        self.en = en
        self.isoc = isoc
        self.cev = cev

    def step(self) -> None:

        # New EV connected  cev = 0 --> 1 ... reset SoC
        if self.cev_prev == 0 and self.cev == 1:
            self.soc = self.isoc

        pabs = min(self.pdel, self.preq)
        self.pabs = pabs
        self.saturated = self.preq <= self.pdel
        self.cev_prev = self.cev

    def reached_saturation(self) -> bool:

        return self.saturated

    def getOutput(self) -> float:

        return self.pabs

    def setSoc(self,
               dt: float = 1) -> None:

        e_prev = self.soc*self.en
        e_now = e_prev + self.pabs*(dt/3600)  # time in hours
        self.soc = min(e_now/self.en, 1)

    def getSoc(self) -> float:
        return self.soc

    def battery_full(self) -> bool:
        return self.soc >= 1
