import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


def stft(signal, fft_size=256, hop_size=128, window_fn=np.hanning):
    signal = np.asarray(signal)

    # Accept complex input: keep dtype
    window = window_fn(fft_size).astype(signal.dtype)

    # Pad signal to make sure we can fit full frames
    pad_width = fft_size - (len(signal) - fft_size) % hop_size
    signal = np.pad(signal, (0, pad_width), mode='constant')

    # Frame the signal
    frames = sliding_window_view(signal, fft_size)[::hop_size]
    windowed_frames = frames * window

    # Complex FFT (supports real or complex input)
    stft_result = np.fft.fft(windowed_frames, n=fft_size, axis=-1)
    return stft_result.T  # shape: (freq_bins, time_frames)

def plot_stft(stft_matrix, sample_rate, fft_size, hop_size):
    time_bins = stft_matrix.shape[1]
    time = np.arange(time_bins) * hop_size / sample_rate
    freq = np.fft.fftfreq(fft_size, d=1/sample_rate)

    # Shift freq and spectrum to center DC (optional for complex signals)
    stft_matrix = np.fft.fftshift(stft_matrix, axes=0)
    freq = np.fft.fftshift(freq)

    magnitude = 20 * np.log10(np.abs(stft_matrix) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(time, freq, magnitude, shading='auto', cmap='magma')
    plt.colorbar(label='Amplitude [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('STFT Magnitude (complex input supported)')
    plt.tight_layout()
    plt.show()

# fs = 8000  # sample rate
# t = np.linspace(0, 1.0, fs, endpoint=False)
# signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
#
# S = stft(signal, fft_size=512, hop_size=128)
# plot_stft(S, sample_rate=fs, fft_size=512, hop_size=128)
