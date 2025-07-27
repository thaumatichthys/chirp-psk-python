import numpy as np
import matplotlib.pyplot as plt


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
