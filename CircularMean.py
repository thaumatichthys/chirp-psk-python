import numpy as np
from misc import *

class CircularMean:
    def __init__(self, max_dphi=1000000, tap=taps_baseband, k=1):
        self.prev_phase = 0
        self.filter = StreamingFIR(taps_baseband)
        self.max_dphi = max_dphi
        self.k = k
    def PushValue(self, complex_value):
        # phase coherence measurement
        phase = np.angle(complex_value)
        dphi = (phase - self.prev_phase + np.pi) % (2 * np.pi) - np.pi

        # dphi = ((dphi + self.max_dphi / 2) % self.max_dphi) - self.max_dphi / 2
        dphi %= self.max_dphi

        self.prev_phase = phase

        uphasor = np.exp(self.k * 1j * dphi)

        return self.filter.pushValue(uphasor), dphi
