import numpy as np
from misc import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
from parameters import *


print(f"Carrier sample rate = {CARRIER_SAMPLERATE}\nChirp BW = {CHIRP_BW_BASEBAND_SAMPLERATE}")

input_text = ("~~~~~~~~~~~~~~~~~~~a wrinkle in falkland by margaret thatcher, an account of britdain, an extremely dank collection of events" +
               "\nDid you know that if you eat rocks, it will taste very bad? I did not, personally.")

input_text = "~~~the quick brown fox jumps over the lazy dog."

data_input = np.unpackbits(np.frombuffer(input_text.encode("utf-8"), dtype=np.uint8)) # * 0 + 1
data_length = len(data_input)



# generate base chirp
n_baseband = np.arange(0, samples_per_symbol_baseband)
baseband_phase = np.pi * (n_baseband * n_baseband / samples_per_symbol_baseband - n_baseband)
baseband_chirp = np.exp(1j * baseband_phase)


# generate carrier (all in one)
n_carrier = np.arange(0, samples_per_symbol_carrier * data_length)
carrier_complex = np.exp(2j * np.pi * CARRIER_CENTER * n_carrier / CARRIER_SAMPLERATE)

# oversample the baseband by zero stuffing
oversampled = np.zeros(len(baseband_chirp) * OVERSAMPLE_RATIO, dtype=baseband_chirp.dtype)
oversampled[::OVERSAMPLE_RATIO] = baseband_chirp

# tiled = np.tile(oversampled, data_length)
# incorporate data into chirps
tiled_chunks = []
for i in range(data_length):
    data = data_input[i]
    chunk = oversampled
    if data:
        chunk = np.conj(chunk)
    tiled_chunks.append(chunk)

tiled = np.concatenate(tiled_chunks)


# 3. Filter the signal
filtered_signal = apply_fir(tiled, taps)  # this is baseband

print(len(filtered_signal))
upconverted_signal = (filtered_signal * carrier_complex).imag

signal_out = upconverted_signal / np.max(np.abs(upconverted_signal))

signal_out += (np.random.random(len(signal_out)) - 0.5)  * 0 * 8


wavfile.write("output.wav", CARRIER_SAMPLERATE, signal_out)

plt.plot(np.abs(np.fft.rfft(signal_out)))
# plt.plot(signal_out)
plt.show()
