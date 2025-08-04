import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from collections import deque
from scipy.signal import firwin, freqz

class StreamingFIR:
    def __init__(self, taps):
        """
        taps: 1D numpy array of FIR coefficients
        """
        self.taps = taps
        self.N = len(taps)
        # delay line initialized with zeros
        self.buffer = deque([0.0]*self.N, maxlen=self.N)

    def pushValue(self, sample):
        """
        Push a new sample in and return one filtered output.
        """
        # append new sample, auto-drops oldest
        self.buffer.append(sample)
        # convert to array for dot product
        buf_arr = np.array(self.buffer)
        # compute y[n] = sum taps[k] * x[n-k]
        return (np.dot(self.taps, buf_arr[::-1]))

def srrc_filter(alpha, sps, num_symbols):
    """
    Generate Square-Root Raised Cosine (SRRC) filter taps.

    Parameters:
    - alpha: Roll-off factor (0 < alpha <= 1)
    - sps: Samples per symbol (integer)
    - num_symbols: Filter span on each side of zero (in symbols)

    Returns:
    - h: SRRC impulse response
    """
    T = 1.0
    t = np.arange(-num_symbols * T, num_symbols * T + 1 / sps, 1 / sps)
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = (1 + alpha * (4 / np.pi - 1)) / np.sqrt(T)
        elif np.isclose(abs(ti), T / (4 * alpha)):
            h[i] = (alpha / np.sqrt(2 * T) *
                    ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                     (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))))
        else:
            num = (np.sin(np.pi * ti * (1 - alpha) / T) +
                   4 * alpha * ti / T * np.cos(np.pi * ti * (1 + alpha) / T))
            den = (np.pi * ti * (1 - (4 * alpha * ti / T) ** 2) / T)
            h[i] = num / den / np.sqrt(T)
    h /= np.sqrt(np.sum(h ** 2))
    return h
def rc_filter(alpha, sps, num_symbols):
    """
    Generate Raised‑Cosine (RC) filter taps.

    Parameters:
    - alpha: Roll‑off factor (0 < alpha <= 1)
    - sps:   Samples per symbol (integer)
    - num_symbols: Filter span on each side of zero (in symbols)

    Returns:
    - h: 1D numpy array of RC impulse response taps
    """
    T = 1.0
    # time axis
    t = np.arange(-num_symbols * T, num_symbols * T + 1/sps, 1/sps)
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        x = ti / T
        denom = 1 - (2 * alpha * x)**2

        # handle t = 0
        if np.isclose(x, 0.0):
            h[i] = 1.0

        # handle denominator = 0 --> t = ±T/(2α)
        elif np.isclose(abs(2 * alpha * x), 1.0):
            # L'Hôpital limit at the singularity
            h[i] = (np.pi / 4) * np.sinc(1/(2*alpha))

        else:
            # sinc term times cosine roll‑off, over the denom
            h[i] = np.sinc(x) * np.cos(np.pi * alpha * x) / denom

    # Normalize for unity DC gain (sum of taps = 1)
    h /= np.sum(h)
    return h

def apply_fir(samples, taps):
    """
    Apply an FIR filter to a sequence of samples.

    Parameters:
    - samples: 1D numpy array of input samples
    - taps: 1D numpy array of FIR filter taps

    Returns:
    - filtered: 1D numpy array of filtered output (same length as input)
    """
    # 'same' mode to keep output length = input length
    return np.convolve(samples, taps, mode='same')


def design_fir_passband_taps(fs, pass_center, pass_width, num_taps, window='hann'):
    """
    Design a linear-phase FIR passband filter using windowed-sinc.

    Parameters:
    - fs            : Sampling rate (Hz)
    - pass_center   : Center frequency of the passband (Hz)
    - pass_width    : Width of the passband (Hz)
    - num_taps      : Number of FIR taps (odd for symmetry; auto-adjusted if even)
    - window        : Window type (string, e.g., 'hamming', 'blackman', 'kaiser')

    Returns:
    - taps          : 1D numpy array of FIR coefficients (real, symmetric)
    """
    # Ensure odd length for true linear phase
    if num_taps % 2 == 0:
        num_taps += 1

    # Compute passband edges
    f1 = pass_center - pass_width / 2.0
    f2 = pass_center + pass_width / 2.0
    # Normalize to Nyquist (fs/2)
    edges = [f1 / (fs/2), f2 / (fs/2)]

    # Design bandpass filter
    taps = firwin(num_taps, edges, window=window, pass_zero=False)
    return taps

# SRRC for RF pulse shaping
alpha = 0.25  # Roll-off factor
span = 6  # Filter span in symbols
taps = srrc_filter(alpha, (samples_per_symbol_carrier / CHIRP_BW_COEFF), span)


# RC for symbol low pass filtering (RC instead of LPF for ISI reduction)
taps_baseband = rc_filter(0.8, CHIRP_BW_COEFF, 2)

# RC for symbol low pass filtering, but for decoder high SR
taps_baseband_2 = rc_filter(0.8, samples_per_symbol_carrier, 2)

# FIR filter for input filtering
taps_input = design_fir_passband_taps(CARRIER_SAMPLERATE, CARRIER_CENTER, CHIRP_BW_BASEBAND_SAMPLERATE * (1 + alpha), 31)

def fold_spectrum(spectrum, decim_factor):
    """
    Fold a complex FFT spectrum to simulate aliasing from integer decimation.

    Parameters:
    - spectrum: 1D complex numpy array (FFT result)
    - decim_factor: integer decimation factor (R)

    Returns:
    - folded_spectrum: aliased spectrum as if the original signal was decimated by R
    """
    spectrum = np.asarray(spectrum)
    N = len(spectrum)
    if N % decim_factor != 0:
        raise ValueError("Length of spectrum must be divisible by decimation factor")

    N_folded = N // decim_factor
    folded = np.zeros(N_folded, dtype=complex)

    for i in range(decim_factor):
        folded += spectrum[i*N_folded:(i+1)*N_folded]

    return folded

# # Example usage:
#
# # 1. Generate SRRC filter taps
# alpha = 0.35  # Roll-off factor
# sps = 8  # Samples per symbol (defines time resolution)
# span = 6  # Filter span in symbols
# taps = srrc_filter(alpha, sps, span)
#
# # 2. Create an example baseband signal: e.g., sine wave + noise
# fs = 8000  # sample rate in Hz
# duration = 0.02  # seconds
# t = np.arange(0, duration, 1 / fs)
# freq = 1000  # Hz tone
# signal = np.sin(2 * np.pi * freq * t) + 0.2 * np.random.randn(len(t))
#
# # 3. Filter the signal
# filtered_signal = apply_fir(signal, taps)
#
# # 4. Plot the original and filtered signals
# plt.figure()
# plt.plot(t, signal, label='Original')
# plt.plot(t, filtered_signal, label='Filtered')
# plt.title("Baseband Signal Before & After SRRC FIR")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend()
# plt.show()
