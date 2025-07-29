import numpy as np
from parameters import *
from scipy.io import wavfile
import matplotlib.pyplot as plt
from misc import *
from clockrecovery import *
from text_decoder import *
from CircularMean import *

samplerate, data = wavfile.read("chirpss-10.wav")

duration = len(data) / samplerate

times = np.linspace(0, duration, len(data))

i_filter = StreamingFIR(taps)
q_filter = StreamingFIR(taps)

baseband_filter_up = StreamingFIR(taps_baseband)
baseband_filter_down = StreamingFIR(taps_baseband)

clockrecovery = ClockRecovery(CHIRP_BW_BASEBAND_SAMPLERATE, DATA_BITRATE, 2)

dummy = []
dummy2 = []
dumm3 = []
dummy4 = []

up_coherence = CircularMean()
down_coherence = CircularMean()

output_bits = []

n_baseband = 0

random_offset = 1

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

        baseband_phase = np.pi * (n_baseband * n_baseband / samples_per_symbol_baseband - n_baseband)
        baseband_chirp_up = np.exp(1j * baseband_phase)
        baseband_chirp_down = -np.conj(baseband_chirp_up)

        dechirp_up = baseband * baseband_chirp_up
        dechirp_down = baseband * baseband_chirp_down

        up_coh = np.abs(up_coherence.PushValue(dechirp_up))
        down_coh = np.abs(down_coherence.PushValue(dechirp_down))

        demodulated = np.sign(up_coh - down_coh)

        clock_pulses = clockrecovery.PushValue(demodulated)

        if clock_pulses:
            output_bits.append(int((demodulated / 2 + 1)))

        dummy4.append(clockrecovery.dummy_ - 1.1)

        dummy.append(up_coh)
        dummy2.append(down_coh)
        dumm3.append(demodulated + 2)

print(output_bits)
info, result_bytes = try_alignments(output_bits)
print("Best alignment info:")
print(info)

plt.plot(dummy)
plt.plot(dummy2)
plt.plot(dumm3)
plt.plot(dummy4)
# plt.plot(np.abs(np.fft.fft(dummy)))
plt.show()
