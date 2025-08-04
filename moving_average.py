import numpy as np


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []
        self.sum = None
        self.index = 0

    def pushValue(self, new_value):
        new_value = np.asarray(new_value)

        if self.sum is None:
            self.sum = np.zeros_like(new_value)

        if len(self.buffer) < self.window_size:
            self.buffer.append(new_value)
            self.sum += new_value
        else:
            old_value = self.buffer[self.index]
            self.sum += new_value - old_value
            self.buffer[self.index] = new_value

        self.index = (self.index + 1) % self.window_size
        return self.sum / len(self.buffer)
