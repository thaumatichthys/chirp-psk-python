import numpy as np
from misc import *

class CircularMean:
    def __init__(self):
        self.prev_phase = 0
        self.filter = StreamingFIR(taps_baseband)
    def PushValue(self, complex_value):
        # phase coherence measurement
        phase = np.angle(complex_value)
        dphi = (phase - self.prev_phase + np.pi) % (2 * np.pi) - np.pi
        self.prev_phase = phase

        uphasor = np.exp(1j * dphi)

        return self.filter.pushValue(uphasor)
