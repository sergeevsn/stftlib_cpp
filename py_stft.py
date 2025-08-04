import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# === Parameters ===
input_file = "data/input.bin"
output_file = "data/output_py.bin"

n_traces = 220
n_samples = 501
frame_size = 64
hop_size = 32
dt = 0.004  # second

# === Load binary file ===
data = np.fromfile(input_file, dtype=np.float32).reshape(n_traces, n_samples)
print(f"Loaded: {data.shape[0]} traces × {data.shape[1]} samples")

# === STFT with padding ===
f, t, Zxx = sps.stft(
    data,
    fs=1/dt,
    nperseg=frame_size,
    noverlap=frame_size - hop_size,
    window='hann',
    axis=-1,
    return_onesided=True,
    boundary='zeros',  # ← will add zero-padding at start/end
    padded=True         # ← will allow returning more frames
)

print("\nTime axis (s):", np.round(t, 4))
print("Frequency axis (Hz):", np.round(f, 2))

# === Inverse STFT ===
_, reconstructed = sps.istft(
    Zxx,
    fs=1/dt,
    nperseg=frame_size,
    noverlap=frame_size - hop_size,
    window='hann',
    input_onesided=True,
    time_axis=2  # Note: time is the last axis
)

print("Reconstructed shape before trim:", reconstructed.shape)

# === Trim to original size and save ===
reconstructed = reconstructed[:, :n_samples].astype(np.float32)
print("Reconstructed shape after trim:", reconstructed.shape)

reconstructed.tofile(output_file)
print(f"Saved reconstructed data to: {output_file}")
