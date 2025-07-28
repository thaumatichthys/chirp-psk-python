DATA_BITRATE = 60

OVERSAMPLE_RATIO = 8  # must be integer
CHIRP_BW_COEFF = 131  # basically chipping rate
CHIRP_BW_BASEBAND_SAMPLERATE = DATA_BITRATE * CHIRP_BW_COEFF  # must be integer fraction of carrier samplerate (also equals baseband samplerate)
CARRIER_SAMPLERATE = OVERSAMPLE_RATIO * CHIRP_BW_BASEBAND_SAMPLERATE  # must be integer multiple of data bitrate
CARRIER_CENTER = 8000
RX_CARRIER_CENTER = 8000

# FFT_INTERVAL_BASEBAND_SAMPLES = CHIRP_BW_BASEBAND_SAMPLERATE / DATA_BITRATE  # how many baseband samples before it does each fft


samples_per_symbol_carrier = int(CARRIER_SAMPLERATE / DATA_BITRATE)
samples_per_symbol_baseband = int(CHIRP_BW_BASEBAND_SAMPLERATE / DATA_BITRATE)
