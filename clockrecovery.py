from iirfilter import *


class ClockRecovery:
    def __init__(self, sample_rate, data_rate, max_dev):
        self.filter_ = IIR_BPF(3, sample_rate, data_rate - max_dev, data_rate + max_dev)
        self.prev_1_ = -1
        self.prev_2_ = 0
        self.prev_3_ = -1

        self.dummy_ = 0
    def PushValue(self, value):
        squared = np.sign(value)
        pulses = np.abs(squared - self.prev_1_)
        self.prev_1_ = squared
        filtered = self.filter_.pushValue(pulses)
        phaseshifted = filtered - self.prev_2_
        self.prev_2_ = filtered
        clock_pulses = np.sign(phaseshifted) - self.prev_3_
        self.prev_3_ = np.sign(phaseshifted)

        self.dummy_ = filtered

        if clock_pulses > 0:
            return 1
        return 0


