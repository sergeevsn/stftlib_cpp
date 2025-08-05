import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

# === Parameters ===
input_file = "data/input.bin"
output_file = "data/output_py.bin"
stft_output_file = "data/stft_py.bin"

n_traces = 220
n_samples = 501
frame_size = 64
hop_size = 32
dt = 0.004  # second

# === Load binary file ===
data = np.fromfile(input_file, dtype=np.float32).reshape(n_traces, n_samples)
print(f"Loaded: {data.shape[0]} traces × {data.shape[1]} samples")

# === STFT with padding ===
# For 2D input with axis=-1, Scipy returns shape (n_traces, n_freqs, n_frames)
f, t, Zxx = sps.stft(
    data,
    fs=1/dt,
    nperseg=frame_size,
    noverlap=frame_size - hop_size,
    window='hann',
    axis=-1,
    return_onesided=True,
    boundary='even',  
    padded=True      
)

print("\nTime axis (s):", np.round(t, 4))
print("Frequency axis (Hz):", np.round(f, 2))
print(f"STFT shape from Scipy: {Zxx.shape} (traces, freqs, frames)")
print(f"Number of frequencies: {Zxx.shape[1]}")
print(f"Number of frames: {Zxx.shape[2]}")


# === Save STFT data to file ===
# Zxx has shape (traces, freqs, frames) - need to convert to C++ format (traces, frames, freqs)
# Original axes: 0=traces, 1=freqs, 2=frames
# Target axes:   0=traces, 1=frames, 2=freqs
stft_data = Zxx.transpose(0, 2, 1)

print(f"\nTransposed STFT shape for saving: {stft_data.shape} (traces, frames, freqs)")

# Write data to file
with open(stft_output_file, 'wb') as f:
    n_traces_out, n_frames_out, n_freqs_out = stft_data.shape
    f.write(np.array([n_traces_out, n_frames_out, n_freqs_out], dtype=np.int32).tobytes())
    
    # Write data as alternating real/imaginary parts
    interleaved_data = np.empty((stft_data.size, 2), dtype=np.float32)
    interleaved_data[:, 0] = stft_data.real.flatten()
    interleaved_data[:, 1] = stft_data.imag.flatten()
    f.write(interleaved_data.tobytes())

print(f"STFT data written to: {stft_output_file}")


# === Inverse STFT ===
# istft requires Zxx in its original format (traces, freqs, frames)
_, reconstructed = sps.istft(
    Zxx,
    fs=1/dt,
    nperseg=frame_size,
    noverlap=frame_size - hop_size,
    window='hann',
    input_onesided=True,
    time_axis=2,      # Time axis is last (index 2)
    freq_axis=1       # Frequency axis is middle (index 1)
)

print("\nReconstructed shape before trim:", reconstructed.shape)

# === Trim to original size and save ===
reconstructed = reconstructed[:, :n_samples].astype(np.float32)
print("Reconstructed shape after trim:", reconstructed.shape)
reconstructed.tofile(output_file)
print(f"Saved reconstructed data to: {output_file}")

# === STFT SPECTRUM ANALYSIS ===
print("\n=== STFT SPECTRUM ANALYSIS ===")

# Calculate statistics for all traces
magnitudes = np.abs(stft_data)
phases = np.angle(stft_data)

print("STFT Magnitude Statistics:")
print(f"  Min: {np.min(magnitudes):.10f}")
print(f"  Max: {np.max(magnitudes):.10f}")
print(f"  Mean: {np.mean(magnitudes):.10f}")
print(f"  Total samples: {magnitudes.size}")

print("STFT Phase Statistics:")
print(f"  Min: {np.min(phases):.6f} rad ({np.min(phases) * 180.0 / np.pi:.2f}°)")
print(f"  Max: {np.max(phases):.6f} rad ({np.max(phases) * 180.0 / np.pi:.2f}°)")
print(f"  Mean: {np.mean(phases):.6f} rad ({np.mean(phases) * 180.0 / np.pi:.2f}°)")

# === RECONSTRUCTED SIGNAL ANALYSIS ===
print("\n=== RECONSTRUCTED SIGNAL ANALYSIS ===")

print("Reconstructed Signal Statistics:")
print(f"  Min: {np.min(reconstructed):.10f}")
print(f"  Max: {np.max(reconstructed):.10f}")
print(f"  Mean: {np.mean(reconstructed):.10f}")
print(f"  Total samples: {reconstructed.size}")

# === RECONSTRUCTION QUALITY ANALYSIS ===
print("\n=== RECONSTRUCTION QUALITY ANALYSIS ===")

# Calculate differences
diff = np.abs(data - reconstructed)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
rms_diff = np.sqrt(np.mean(diff**2))

print("Reconstruction vs Original:")
print(f"  Max absolute difference: {max_diff:.10f}")
print(f"  Mean absolute difference: {mean_diff:.10f}")
print(f"  RMS difference: {rms_diff:.10f}")

# Calculate relative error
mean_original = np.mean(np.abs(data))
relative_error = mean_diff / mean_original * 100.0

print(f"  Relative error: {relative_error:.6f}%")
