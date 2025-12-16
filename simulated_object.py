class SimulatedObject:
    '''
    Class implementing an abstract simulated object
    '''

    def setInput(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s "
                                  f"{self.setInput.__name__} "
                                  "method has not been implemented.")

    def step(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s "
                                  f"{self.step.__name__} "
                                  "method has not been implemented.")

    def getOutput(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s "
                                  f"{self.getOutput.__name__} "
                                  "method has not been implemented.")

    def reached_saturation(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s "
                                  f"{self.reached_saturation.__name__} "
                                  "method has not been implemented.")
