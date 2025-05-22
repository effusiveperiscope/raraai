import numpy as np

# Parameters
n_mels = 128
sample_rate = 44100

# Mel scale conversion (HTK formula)
def mel_to_hz(mels):
    return 700 * (10**(mels / 2595.0) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

# Generate mel points and convert to frequency
mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate // 2), n_mels)
hz_points = mel_to_hz(mel_points)

# Print frequencies for each mel bin
for i, freq in enumerate(hz_points):
    print(f"Mel bin {i:3d}: {freq:8.2f} Hz")