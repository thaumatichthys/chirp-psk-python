DATA_BITRATE = 93.75# 46.875

OVERSAMPLE_RATIO = 8  # must be integer
CHIRP_BW_COEFF = 64# 256  # basically chipping rate
CHIRP_BW_BASEBAND_SAMPLERATE = DATA_BITRATE * CHIRP_BW_COEFF  # must be integer fraction of carrier samplerate (also equals baseband samplerate)
CARRIER_SAMPLERATE = OVERSAMPLE_RATIO * CHIRP_BW_BASEBAND_SAMPLERATE  # must be integer multiple of data bitrate
CARRIER_CENTER = 15000
RX_CARRIER_CENTER = 15000


# CARRIER_SAMPLERATE / (CLOCK_RECOVERY_OVERSAMPLE_RATIO * DATA_BITRATE)
# OSR * DATA_BITRATE * CHIRP_BW_COEFF / (CLOCK_RECOVERY_SR * DATA_BITRATE)
# OSR * CHIRP_BW_COEFF / CLOCK_RECOVERY_SR = samples_per_clk_recovert_symbol
CLOCK_RECOVERY_OVERSAMPLE_RATIO = 32  # must be integer divisor of (OSR * CHIRP_BW_COEFF)
samples_per_clk_recovery_symbol = int(OVERSAMPLE_RATIO * CHIRP_BW_COEFF / CLOCK_RECOVERY_OVERSAMPLE_RATIO)

ACQ_AVERAGES = 24

# FFT_INTERVAL_BASEBAND_SAMPLES = CHIRP_BW_BASEBAND_SAMPLERATE / DATA_BITRATE  # how many baseband samples before it does each fft

# osr * chirp_bw_baseband_samplerate / data_bitrate
# osr * data_bitrate * chirp_bw_coeff / data_bitrate
# osr * chirp_bw_coeff
samples_per_symbol_carrier = int(CARRIER_SAMPLERATE / DATA_BITRATE)
# data_bitrate * chirp_bw_coeff / data_bitrate = chirp_bw_coeff
samples_per_symbol_baseband = int(CHIRP_BW_BASEBAND_SAMPLERATE / DATA_BITRATE)

#samples_per_symbol_carrier / CHIRP_BW_COEFF