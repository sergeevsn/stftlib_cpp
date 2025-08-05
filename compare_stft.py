import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. Parameters and file paths ===
N_TRACES = 220
N_SAMPLES = 501
FRAME_SIZE = 64
HOP_SIZE = 32
DT = 0.004

OUTPUT_DIR = "pics"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CPP_STFT_FILE = "data/stft_cpp.bin"
PY_STFT_FILE = "data/stft_py.bin"
ORIGINAL_FILE = "data/input.bin"
CPP_RECON_FILE = "data/output_cpp.bin"
PY_RECON_FILE = "data/output_py.bin"


# === 2. Data loading functions ===

def load_stft_bin_cpp(path):
    """Loads C++ STFT data (traces, frames, freqs)."""
    with open(path, 'rb') as f:
        n_traces_f, n_frames_f, n_freqs_f = np.frombuffer(f.read(12), dtype=np.int32)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_traces_f, n_frames_f, n_freqs_f, 2)
        return data[..., 0] + 1j * data[..., 1]

def load_stft_bin_py(path):
    """
    Loads Python STFT data that was saved in format (traces, frames, freqs).
    """
    with open(path, 'rb') as f:
        # 1. Read header. Get: 220, 17, 33
        n_traces_f, n_frames_f, n_freqs_f = np.frombuffer(f.read(12), dtype=np.int32)
        print(n_traces_f, n_frames_f, n_freqs_f)
        
        # 2. Read data and restore complex numbers
        raw_data = np.frombuffer(f.read(), dtype=np.float32)
        complex_data = raw_data[::2] + 1j * raw_data[1::2]
        
        # 3. Give array correct shape (220, 17, 33).
        #    No transposition needed.
        return complex_data.reshape(n_traces_f, n_frames_f, n_freqs_f)

def load_seismogram_bin(path, traces, samples):
    """Loads seismogram data."""
    return np.fromfile(path, dtype=np.float32).reshape(traces, samples)


# === 3. Visualization functions ===

def plot_seismogram_comparison(original, recon_cpp, recon_py, output_path):
    print("Creating seismogram comparison plot...")
    diff_cpp = original - recon_cpp
    diff_py = original - recon_py
    vmin, vmax = np.percentile(original, [2, 98])
   
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle('Comparison of Original and Reconstructed Seismograms', fontsize=16)
    axs[0, 0].imshow(original.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 0].set_title('Original')
    axs[0, 0].set_ylabel('Time samples')
    axs[0, 1].imshow(recon_cpp.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 1].set_title('Reconstructed (C++)')
    axs[0, 2].imshow(diff_cpp.T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 2].set_title('Difference (Original - C++)')
    axs[1, 0].imshow(original.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 0].set_ylabel('Time samples')
    axs[1, 0].set_xlabel('Trace number')
    axs[1, 1].imshow(recon_py.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 1].set_title('Reconstructed (Python)')
    axs[1, 1].set_xlabel('Trace number')
    axs[1, 2].imshow(diff_py.T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 2].set_title('Difference (Original - Python)')
    axs[1, 2].set_xlabel('Trace number')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  Plot saved to {output_path}")
    plt.close(fig)

def plot_stft_comparison(stft_cpp, stft_py, f_axis, t_axis, output_path):
    print("Creating STFT comparison plot...")
    freqs_to_plot_hz = [20, 40, 60]
    freq_indices = [np.argmin(np.abs(f_axis - f)) for f in freqs_to_plot_hz]
    fig, axs = plt.subplots(len(freqs_to_plot_hz), 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle('Comparison of STFT Amplitude Spectra', fontsize=16)
    for i, freq_idx in enumerate(freq_indices):
        freq_val = f_axis[freq_idx]
        cpp_slice = np.abs(stft_cpp[:, :, freq_idx])
        py_slice = np.abs(stft_py[:, :, freq_idx])    
        vmin, vmax = np.percentile(cpp_slice, [2, 98])
        axs[i, 0].imshow(cpp_slice.T, aspect='auto', cmap='viridis', origin='upper', extent=[0, N_TRACES, t_axis[0], t_axis[-1]], vmin=vmin, vmax=vmax)
        axs[i, 0].set_ylabel(f'f = {freq_val:.1f} Hz\nTime (s)')
        im = axs[i, 1].imshow(py_slice.T, aspect='auto', cmap='viridis', origin='upper', extent=[0, N_TRACES, t_axis[0], t_axis[-1]], vmin=vmin, vmax=vmax)
       
    axs[0, 0].set_title('C++ STFT')
    axs[0, 1].set_title('Python STFT')
    axs[-1, 0].set_xlabel('Trace number')
    axs[-1, 1].set_xlabel('Trace number')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  Plot saved to {output_path}")
    plt.close(fig)


# === 3.5. Statistical analysis functions ===

def analyze_stft_statistics(stft_cpp, stft_py, f_axis, t_axis):
    """Analyzes statistics of amplitude and phase spectra of STFT."""
    print("\n=== STFT STATISTICS ANALYSIS ===")
    
    # Calculate amplitude and phase spectra
    amp_cpp = np.abs(stft_cpp)
    amp_py = np.abs(stft_py)
    phase_cpp = np.angle(stft_cpp)
    phase_py = np.angle(stft_py)
    
    # Statistics for amplitude spectra
    print("\n--- AMPLITUDE SPECTRA ---")
    print(f"Mean value C++: {np.mean(amp_cpp):.6f}")
    print(f"Mean value Python: {np.mean(amp_py):.6f}")
    print(f"Difference of means: {np.mean(amp_cpp) - np.mean(amp_py):.2e}")
    
    print(f"Standard deviation C++: {np.std(amp_cpp):.6f}")
    print(f"Standard deviation Python: {np.std(amp_py):.6f}")
    print(f"Difference of std: {np.std(amp_cpp) - np.std(amp_py):.2e}")
    
    print(f"Maximum C++: {np.max(amp_cpp):.6f}")
    print(f"Maximum Python: {np.max(amp_py):.6f}")
    print(f"Difference of maxima: {np.max(amp_cpp) - np.max(amp_py):.2e}")
    
    print(f"Minimum C++: {np.min(amp_cpp):.6f}")
    print(f"Minimum Python: {np.min(amp_py):.6f}")
    print(f"Difference of minima: {np.min(amp_cpp) - np.min(amp_py):.2e}")
    
    # Relative error of amplitudes
    rel_error_amp = np.abs(amp_cpp - amp_py) / (np.abs(amp_cpp) + 1e-10)
    print(f"Mean relative error of amplitudes: {np.mean(rel_error_amp):.2e}")
    print(f"Maximum relative error of amplitudes: {np.max(rel_error_amp):.2e}")
    
    # Statistics for phase spectra
    print("\n--- PHASE SPECTRA ---")
    print(f"Mean value C++: {np.mean(phase_cpp):.6f}")
    print(f"Mean value Python: {np.mean(phase_py):.6f}")
    print(f"Difference of means: {np.mean(phase_cpp) - np.mean(phase_py):.2e}")
    
    print(f"Standard deviation C++: {np.std(phase_cpp):.6f}")
    print(f"Standard deviation Python: {np.std(phase_py):.6f}")
    print(f"Difference of std: {np.std(phase_cpp) - np.std(phase_py):.2e}")
    
    # Absolute phase error (considering cyclic nature)
    phase_diff = np.abs(phase_cpp - phase_py)
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Consider cyclic nature
    print(f"Mean absolute phase error: {np.mean(phase_diff):.2e}")
    print(f"Maximum absolute phase error: {np.max(phase_diff):.2e}")
    
    # Statistics by frequencies
    print("\n--- FREQUENCY STATISTICS ---")
    for i, freq in enumerate([20, 40, 60]):
        freq_idx = np.argmin(np.abs(f_axis - freq))
        amp_cpp_freq = amp_cpp[:, :, freq_idx]
        amp_py_freq = amp_py[:, :, freq_idx]
        
        rel_error_freq = np.abs(amp_cpp_freq - amp_py_freq) / (np.abs(amp_cpp_freq) + 1e-10)
        print(f"Frequency {freq} Hz - mean relative error: {np.mean(rel_error_freq):.2e}")
    
    return {
        'amp_cpp': amp_cpp,
        'amp_py': amp_py,
        'phase_cpp': phase_cpp,
        'phase_py': phase_py,
        'rel_error_amp': rel_error_amp,
        'phase_diff': phase_diff
    }

def analyze_seismogram_statistics(original, recon_cpp, recon_py):
    """Analyzes statistics of amplitudes of reconstructed seismograms."""
    print("\n=== SEISMOGRAM STATISTICS ANALYSIS ===")
    
    # Statistics for amplitudes
    print("\n--- SEISMOGRAM AMPLITUDES ---")
    print(f"Mean value original: {np.mean(original):.6f}")
    print(f"Mean value C++: {np.mean(recon_cpp):.6f}")
    print(f"Mean value Python: {np.mean(recon_py):.6f}")
    
    print(f"Standard deviation original: {np.std(original):.6f}")
    print(f"Standard deviation C++: {np.std(recon_cpp):.6f}")
    print(f"Standard deviation Python: {np.std(recon_py):.6f}")
    
    print(f"Maximum original: {np.max(original):.6f}")
    print(f"Maximum C++: {np.max(recon_cpp):.6f}")
    print(f"Maximum Python: {np.max(recon_py):.6f}")
    
    print(f"Minimum original: {np.min(original):.6f}")
    print(f"Minimum C++: {np.min(recon_cpp):.6f}")
    print(f"Minimum Python: {np.min(recon_py):.6f}")
    
    # Reconstruction errors
    error_cpp = original - recon_cpp
    error_py = original - recon_py
    
    print(f"\n--- RECONSTRUCTION ERRORS ---")
    print(f"Mean absolute error C++: {np.mean(np.abs(error_cpp)):.6f}")
    print(f"Mean absolute error Python: {np.mean(np.abs(error_py)):.6f}")
    
    print(f"Root mean square error C++: {np.sqrt(np.mean(error_cpp**2)):.6f}")
    print(f"Root mean square error Python: {np.sqrt(np.mean(error_py**2)):.6f}")
    
    # Relative errors
    rel_error_cpp = np.abs(error_cpp) / (np.abs(original) + 1e-10)
    rel_error_py = np.abs(error_py) / (np.abs(original) + 1e-10)
    
    print(f"Mean relative error C++: {np.mean(rel_error_cpp):.2e}")
    print(f"Mean relative error Python: {np.mean(rel_error_py):.2e}")
    
    print(f"Maximum relative error C++: {np.max(rel_error_cpp):.2e}")
    print(f"Maximum relative error Python: {np.max(rel_error_py):.2e}")
    
    # Comparison between C++ and Python reconstructions
    diff_cpp_py = recon_cpp - recon_py
    print(f"\n--- C++ vs PYTHON COMPARISON ---")
    print(f"Mean difference C++ - Python: {np.mean(diff_cpp_py):.2e}")
    print(f"Standard deviation of difference: {np.std(diff_cpp_py):.2e}")
    print(f"Maximum absolute difference: {np.max(np.abs(diff_cpp_py)):.2e}")
    
    rel_diff_cpp_py = np.abs(diff_cpp_py) / (np.abs(recon_cpp) + 1e-10)
    print(f"Mean relative difference C++ vs Python: {np.mean(rel_diff_cpp_py):.2e}")
    print(f"Maximum relative difference C++ vs Python: {np.max(rel_diff_cpp_py):.2e}")
    
    return {
        'error_cpp': error_cpp,
        'error_py': error_py,
        'rel_error_cpp': rel_error_cpp,
        'rel_error_py': rel_error_py,
        'diff_cpp_py': diff_cpp_py,
        'rel_diff_cpp_py': rel_diff_cpp_py
    }

def plot_statistics_histograms(stft_stats, seis_stats, output_path):
    """Creates histograms of statistical distributions."""
    print("Creating statistical distribution histograms...")
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Distributions', fontsize=16)
    
    # Amplitude spectra
    axs[0, 0].hist(stft_stats['amp_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[0, 0].hist(stft_stats['amp_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[0, 0].set_title('STFT Amplitude Distribution')
    axs[0, 0].set_xlabel('Amplitude')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')
    
    # Phase spectra
    axs[0, 1].hist(stft_stats['phase_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[0, 1].hist(stft_stats['phase_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[0, 1].set_title('STFT Phase Distribution')
    axs[0, 1].set_xlabel('Phase (rad)')
    axs[0, 1].set_ylabel('Density')
    axs[0, 1].legend()
    
    # Relative amplitude errors
    axs[0, 2].hist(stft_stats['rel_error_amp'].flatten(), bins=50, alpha=0.7, density=True)
    axs[0, 2].set_title('Relative Amplitude Error Distribution')
    axs[0, 2].set_xlabel('Relative Error')
    axs[0, 2].set_ylabel('Density')
    axs[0, 2].set_yscale('log')
    
    # Reconstruction errors
    axs[1, 0].hist(seis_stats['error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[1, 0].hist(seis_stats['error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 0].set_title('Reconstruction Error Distribution')
    axs[1, 0].set_xlabel('Error')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()
    
    # Relative reconstruction errors
    axs[1, 1].hist(seis_stats['rel_error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[1, 1].hist(seis_stats['rel_error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 1].set_title('Relative Reconstruction Error Distribution')
    axs[1, 1].set_xlabel('Relative Error')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    
    # Difference between C++ and Python
    axs[1, 2].hist(seis_stats['diff_cpp_py'].flatten(), bins=50, alpha=0.7, density=True)
    axs[1, 2].set_title('C++ - Python Difference Distribution')
    axs[1, 2].set_xlabel('Difference')
    axs[1, 2].set_ylabel('Density')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  Histograms saved to {output_path}")
    plt.close(fig)


# === 4. Main execution block ===
if __name__ == "__main__":
    print("Loading data...")
    stft_cpp = load_stft_bin_cpp(CPP_STFT_FILE)
    stft_py = load_stft_bin_py(PY_STFT_FILE)
    original_seis = load_seismogram_bin(ORIGINAL_FILE, N_TRACES, N_SAMPLES)
    recon_cpp = load_seismogram_bin(CPP_RECON_FILE, N_TRACES, N_SAMPLES)
    recon_py = load_seismogram_bin(PY_RECON_FILE, N_TRACES, N_SAMPLES)

    print(f"STFT C++ shape: {stft_cpp.shape}")
    print(f"STFT Python shape: {stft_py.shape}")
    if stft_cpp.shape != stft_py.shape:
        print("\n[ERROR] STFT array shapes do not match! Check loading functions.")
        exit()

    n_frames = stft_cpp.shape[1]
    n_freqs = stft_cpp.shape[2]
    t_axis = np.arange(n_frames) * HOP_SIZE * DT
    f_axis = np.arange(n_freqs) / (DT * FRAME_SIZE)

    plot_seismogram_comparison(
        original=original_seis,
        recon_cpp=recon_cpp,
        recon_py=recon_py,
        output_path=os.path.join(OUTPUT_DIR, 'comparison_seismograms.png')
    )
    plot_stft_comparison(
        stft_cpp=stft_cpp,
        stft_py=stft_py,
        f_axis=f_axis,
        t_axis=t_axis,
        output_path=os.path.join(OUTPUT_DIR, 'comparison_stft_frames.png')
    )

    # Statistical analysis
    stft_stats = analyze_stft_statistics(stft_cpp, stft_py, f_axis, t_axis)
    seis_stats = analyze_seismogram_statistics(original_seis, recon_cpp, recon_py)
    plot_statistics_histograms(stft_stats, seis_stats, os.path.join(OUTPUT_DIR, 'statistics_histograms.png'))

    print("\nAnalysis completed.")