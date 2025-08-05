import numpy as np
from parameters import *
from scipy.io import wavfile
import matplotlib.pyplot as plt
from misc import *
from clockrecovery import *
from text_decoder import *
from CircularMean import *
from stft import *
from moving_average import *

samplerate, data = wavfile.read("output.wav")

duration = len(data) / samplerate

times = np.linspace(0, duration, len(data))

i_filter = StreamingFIR(taps)
q_filter = StreamingFIR(taps)

baseband_filter_up = StreamingFIR(taps_baseband)
baseband_filter_down = StreamingFIR(taps_baseband)

input_filter = StreamingFIR(taps_input)

clockrecovery = ClockRecovery(DATA_BITRATE * CLOCK_RECOVERY_OVERSAMPLE_RATIO, DATA_BITRATE, 1)

dummy = []
dummy2 = []
dumm3 = []
dummy4 = []
dummy5 = []

up_coherence = CircularMean(2 * np.pi / OVERSAMPLE_RATIO, taps_baseband_2, 8)
down_coherence = CircularMean(2 * np.pi / OVERSAMPLE_RATIO, taps_baseband_2, 8)

output_bits = []

n_baseband = 0
n_baseband_corr = 0
#acq_moving_avg = MovingAverage(ACQ_AVERAGES)
acq_avg = np.zeros(int(samples_per_symbol_baseband / 2))
acq_avg_counter = 0

random_offset = 1234

up_fft_buf = []
down_fft_buf = []

demodulated = 1

for i in range(len(times) - random_offset):
    # giant loop
    t = times[i]

    input = data[i + random_offset]

    filtered_input = input_filter.pushValue(input)
    #dummy5.append(filtered_input)
    # filtered_input = input

    # downconvert to baseband
    lo_complex = np.exp(2j * np.pi * t * RX_CARRIER_CENTER)

    i_baseband = i_filter.pushValue(lo_complex.real * filtered_input)
    q_baseband = q_filter.pushValue(lo_complex.imag * filtered_input)


    # run at baseband rate


    # n_baseband = (i % samples_per_symbol_carrier) / OVERSAMPLE_RATIO + n_baseband_corr
    n_baseband = ((i + n_baseband_corr) % samples_per_symbol_carrier) / OVERSAMPLE_RATIO

    baseband = i_baseband + 1j * q_baseband

    baseband_phase = np.pi * (n_baseband * n_baseband / samples_per_symbol_baseband - n_baseband)
    baseband_chirp_up = np.exp(1j * baseband_phase)
    baseband_chirp_down = -np.conj(baseband_chirp_up)

    dechirp_up = baseband * baseband_chirp_up
    dechirp_down = baseband * baseband_chirp_down

    down_fft_buf.append(dechirp_down)
    up_fft_buf.append(dechirp_up)

    if i == 0:
        up_fft_buf = []
        down_fft_buf = []
    # if n_baseband % samples_per_symbol_baseband == 0 and i > 0:
    if (i + n_baseband_corr) % samples_per_symbol_carrier == 0 and i > 0:
        if len(down_fft_buf) == samples_per_symbol_carrier:
            down_fft = fold_spectrum(np.abs(np.fft.fft(down_fft_buf) ** 2), OVERSAMPLE_RATIO)
            up_fft = fold_spectrum(np.abs(np.fft.fft(up_fft_buf) ** 2), OVERSAMPLE_RATIO)

            up_folded = up_fft[:int(len(up_fft) / 2)] + up_fft[int(len(up_fft) / 2):][::-1]
            down_folded = down_fft[:int(len(down_fft) / 2)] + down_fft[int(len(down_fft) / 2):][::-1]

            acq_avg += np.float64(down_folded + up_folded)
            print(len(acq_avg))
            if acq_avg_counter % ACQ_AVERAGES == 0:
                index = np.argmax(acq_avg)

                # index = np.argmax(up_folded + down_folded)
                if index > 0.1:
                    print(f"UNLOCK {index}")
                else:
                    print("LOCKED")
                # print(f"CORRECTION {index}")
                n_baseband_corr += index * OVERSAMPLE_RATIO
                acq_avg *= 0
            acq_avg_counter += 1

            demodulated = up_folded[0] - down_folded[0]
            demodulated += up_folded[1] - down_folded[1]
            # avg = acq_moving_avg.pushValue(down_folded + up_folded)


            # summed = np.abs(up_fft + down_fft)
            # peak_ind = np.argmax(summed)
            # peak_val = summed[peak_ind]

            # plt.plot(up_fft_buf)


            # demodulated = np.max(up_fft) - np.max(down_fft)

            # plt.plot(up_folded + down_folded)
            # plt.plot(down_folded, marker='o')

            # plt.show()
        down_fft_buf = []
        up_fft_buf = []
    # up_coh_raw, dphi_up = up_coherence.PushValue(dechirp_up)
    # down_coh_raw, dphi_down = down_coherence.PushValue(dechirp_down)

    if i % samples_per_clk_recovery_symbol == 0:
        # up_coh = np.abs(up_coh_raw)
       #  down_coh = np.abs(down_coh_raw)

        # demodulated = 1 # np.sign(up_coh - down_coh)

        clock_pulses = clockrecovery.PushValue(demodulated)

        if clock_pulses:
            output_bits.append(int((np.sign(demodulated) / 2 + 1)))

    #dummy4.append(clockrecovery.dummy_ - 1.1)

    dummy.append(baseband_chirp_up + 3)
    dummy2.append(baseband.imag)
    dumm3.append(dechirp_up - 3)
    # dummy.append(dphi_up + 1)
    # dummy2.append(dphi_down - 1)
    # dummy4.append(dphi_up % (2 * np.pi / OVERSAMPLE_RATIO) + 4)
    # dumm3.append(up_coh - down_coh)
    # dummy5.append(filtered_input)
    if i % 2048 == 0:
        print(f"completed {100 * i / (len(data))}%")

print(output_bits)
info, result_bytes = try_alignments(output_bits)
print("Best alignment info:")
print(info)

# S = stft(dummy5, fft_size=2048, hop_size=128)
# plot_stft(S, sample_rate=CARRIER_SAMPLERATE, fft_size=2048, hop_size=128)


plt.plot(dummy)
plt.plot(dummy2)
plt.plot(dumm3)
# plt.plot(dummy4)
# plt.plot(np.abs(np.fft.fft(dummy5)))
# plt.plot(dummy5)
plt.show()
