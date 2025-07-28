import numpy as np
from parameters import *
from scipy.io import wavfile
import matplotlib.pyplot as plt
from misc import *


samplerate, data = wavfile.read("output.wav")

duration = len(data) / samplerate

times = np.linspace(0, duration, len(data))


i_filter = StreamingFIR(taps)
q_filter = StreamingFIR(taps)

baseband_filter_up = StreamingFIR(taps_baseband)
baseband_filter_down = StreamingFIR(taps_baseband)


dummy = []
dummy2 = []
dumm3 = []
n_baseband = 0

fft_buffer = []

random_offset = 234

for i in range(len(times) - random_offset):
    # giant loop
    t = times[i]

    input = data[i + random_offset]

    # downconvert to baseband
    lo_complex = np.exp(2j * np.pi * t * RX_CARRIER_CENTER)

    i_baseband = i_filter.pushValue(lo_complex.real * input)
    q_baseband = q_filter.pushValue(lo_complex.imag * input)

    if i % OVERSAMPLE_RATIO == 0:
        # run at baseband rate
        n_baseband += 1

        baseband = i_baseband + 1j * q_baseband
        # dummy.append(i_baseband)

        baseband_phase = np.pi * (n_baseband * n_baseband / samples_per_symbol_baseband - n_baseband)
        baseband_chirp_up = np.exp(1j * baseband_phase)
        baseband_chirp_down = np.conj(baseband_chirp_up)

        dechirp_up = baseband * baseband_chirp_up
        dechirp_down = baseband * baseband_chirp_down

        fft_buffer.append(dechirp_down + dechirp_up)

        dechirped_up_lowpassed = np.abs(baseband_filter_up.pushValue(dechirp_up))
        dechirped_down_lowpassed = np.abs(baseband_filter_down.pushValue(dechirp_down))

        demodulated = np.sign(dechirped_down_lowpassed - dechirped_up_lowpassed)

        # correct for rotations
        if n_baseband % samples_per_symbol_baseband == 0:
            fftd = np.fft.fft(fft_buffer)
            # print(len(fft_buffer))
            max_index = np.argmax(np.abs(fftd))
            n_baseband += max_index
            if max_index != 0:
                print(f"ADJUSTED: {max_index}")
           # print(max_index)
            fft_buffer = []



        dummy.append(dechirped_up_lowpassed)
        dummy2.append(dechirped_down_lowpassed)
        dumm3.append(demodulated + 2)

plt.plot(dummy)
plt.plot(dummy2)
plt.plot(dumm3)
# plt.plot(np.abs(np.fft.fft(dummy)))
plt.show()






