import numpy as np
from misc import *
import matplotlib.pyplot as plt
from scipy.io import wavfile

DATA_BITRATE = 80

OVERSAMPLE_RATIO = 12  # must be integer
CHIRP_BW_COEFF = 69  # basically chipping rate
CHIRP_BW_BASEBAND_SAMPLERATE = DATA_BITRATE * CHIRP_BW_COEFF  # must be integer fraction of carrier samplerate (also equals baseband samplerate)
CARRIER_SAMPLERATE = OVERSAMPLE_RATIO * CHIRP_BW_BASEBAND_SAMPLERATE  # must be integer multiple of data bitrate
CARRIER_CENTER = 8000

print(f"Carrier sample rate = {CARRIER_SAMPLERATE}\nChirp BW = {CHIRP_BW_BASEBAND_SAMPLERATE}")

input_text = ("~~~~~~~~~~~~~~~~~~~a wrinkle in falkland by margaret thatcher, an account of britdain, an extremely dank collection of events" +
               "\nDid you know that if you eat rocks, it will taste very bad? I did not, personally.")

# input_text = "testtest"

data_input = np.unpackbits(np.frombuffer(input_text.encode("utf-8"), dtype=np.uint8))  # * 0 + 1
data_length = len(data_input)

samples_per_symbol_carrier = int(CARRIER_SAMPLERATE / DATA_BITRATE)
samples_per_symbol_baseband = int(CHIRP_BW_BASEBAND_SAMPLERATE / DATA_BITRATE)

# generate base chirp
n_baseband = np.arange(0, samples_per_symbol_baseband)
baseband_phase = np.pi * (n_baseband * n_baseband / samples_per_symbol_baseband - n_baseband)
baseband_chirp = np.exp(1j * baseband_phase)
print(baseband_chirp)

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
    data = data_input[i] * 2.0 - 1
    tiled_chunks.append(data * oversampled)

tiled = np.concatenate(tiled_chunks)

# 1. Generate SRRC filter taps
alpha = 0.35  # Roll-off factor
span = 6  # Filter span in symbols
taps = srrc_filter(alpha, (samples_per_symbol_carrier / CHIRP_BW_COEFF), span)

# 3. Filter the signal
filtered_signal = apply_fir(tiled, taps)  # this is baseband

print(len(filtered_signal))
upconverted_signal = (filtered_signal * carrier_complex).real


wavfile.write("output.wav", CARRIER_SAMPLERATE, upconverted_signal)

# plt.plot(np.abs(np.fft.fft(upconverted_signal)))
plt.plot(upconverted_signal)
plt.show()
