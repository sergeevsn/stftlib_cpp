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

# Bins/samples below this fraction of peak amplitude are excluded from relative metrics.
SIGNIFICANCE_FRAC = 0.01


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

def _significance_threshold(values, fraction=SIGNIFICANCE_FRAC):
    """Minimum |value| for a sample/bin to count in relative-error statistics."""
    peak = float(np.max(np.abs(values)))
    return max(1e-10, fraction * peak)


def _cyclic_phase_diff(phase_a, phase_b):
    diff = np.abs(phase_a - phase_b)
    return np.minimum(diff, 2 * np.pi - diff)


def _masked_relative_error(diff, reference, threshold):
    """Relative |diff|/|reference| only where |reference| >= threshold."""
    mask = np.abs(reference) >= threshold
    if not np.any(mask):
        return np.array([]), mask
    rel = np.abs(diff)[mask] / np.abs(reference)[mask]
    return rel, mask


def _print_abs_error_block(title, abs_err):
    print(f"\n--- {title} ---")
    print(f"Mean absolute error: {np.mean(abs_err):.6e}")
    print(f"RMS error: {np.sqrt(np.mean(abs_err ** 2)):.6e}")
    print(f"Maximum absolute error: {np.max(abs_err):.6e}")


def _print_rel_error_block(title, rel, mask, threshold):
    n_total = mask.size
    n_used = int(np.count_nonzero(mask))
    print(f"\n--- {title} ---")
    print(f"Significance threshold: {threshold:.6e} ({SIGNIFICANCE_FRAC * 100:.1f}% of peak)")
    print(f"Samples used: {n_used} / {n_total} ({100.0 * n_used / n_total:.1f}%)")
    if rel.size == 0:
        print("No samples above threshold.")
        return
    print(f"Mean relative error: {np.mean(rel):.6e}")
    print(f"Median relative error: {np.median(rel):.6e}")
    print(f"95th percentile relative error: {np.percentile(rel, 95):.6e}")
    print(f"Maximum relative error: {np.max(rel):.6e}")


def analyze_stft_statistics(stft_cpp, stft_py, f_axis, t_axis):
    """Analyzes statistics of amplitude and phase spectra of STFT."""
    print("\n=== STFT STATISTICS ANALYSIS ===")

    amp_cpp = np.abs(stft_cpp)
    amp_py = np.abs(stft_py)
    amp_diff = np.abs(amp_cpp - amp_py)
    complex_diff = np.abs(stft_cpp - stft_py)

    amp_threshold = _significance_threshold(amp_py)
    sig_mask = np.maximum(amp_cpp, amp_py) >= amp_threshold

    rel_amp, _ = _masked_relative_error(amp_diff, amp_py, amp_threshold)

    print("\n--- AMPLITUDE SPECTRA (distributions) ---")
    print(f"Mean C++: {np.mean(amp_cpp):.6f}  |  Mean Python: {np.mean(amp_py):.6f}")
    print(f"Std  C++: {np.std(amp_cpp):.6f}  |  Std  Python: {np.std(amp_py):.6f}")
    print(f"Max  C++: {np.max(amp_cpp):.6f}  |  Max  Python: {np.max(amp_py):.6f}")

    _print_abs_error_block("AMPLITUDE DIFFERENCES (C++ vs Python)", amp_diff)
    _print_abs_error_block("COMPLEX COEFFICIENT DIFFERENCES (C++ vs Python)", complex_diff)
    _print_rel_error_block("RELATIVE AMPLITUDE ERRORS (significant bins only)", rel_amp, sig_mask, amp_threshold)

    print("\n--- CORRELATION (C++ vs Python) ---")
    print(f"Amplitude correlation: {np.corrcoef(amp_cpp.ravel(), amp_py.ravel())[0, 1]:.6f}")
    print(f"Real part correlation: {np.corrcoef(stft_cpp.real.ravel(), stft_py.real.ravel())[0, 1]:.6f}")
    print(f"Imag part correlation: {np.corrcoef(stft_cpp.imag.ravel(), stft_py.imag.ravel())[0, 1]:.6f}")

    phase_cpp = np.angle(stft_cpp)
    phase_py = np.angle(stft_py)
    phase_diff = _cyclic_phase_diff(phase_cpp, phase_py)

    print("\n--- PHASE DIFFERENCES (significant bins only) ---")
    if np.any(sig_mask):
        phase_sig = phase_diff[sig_mask]
        print(f"Mean absolute phase error: {np.mean(phase_sig):.6e} rad ({np.degrees(np.mean(phase_sig)):.2f}°)")
        print(f"Median absolute phase error: {np.median(phase_sig):.6e} rad")
        print(f"95th percentile phase error: {np.percentile(phase_sig, 95):.6e} rad")
        print(f"Maximum absolute phase error: {np.max(phase_sig):.6e} rad")
    else:
        phase_sig = np.array([])
        print("No samples above threshold.")

    print("\n--- FREQUENCY SLICES (significant bins only) ---")
    for freq in [20, 40, 60]:
        freq_idx = np.argmin(np.abs(f_axis - freq))
        slice_diff = amp_diff[:, :, freq_idx]
        slice_py = amp_py[:, :, freq_idx]
        slice_rel, slice_mask = _masked_relative_error(slice_diff, slice_py, amp_threshold)
        if slice_rel.size:
            print(
                f"f = {freq:2d} Hz: mean rel = {np.mean(slice_rel):.4e}, "
                f"max abs = {np.max(slice_diff):.4e}, "
                f"bins used = {np.count_nonzero(slice_mask)}/{slice_mask.size}"
            )
        else:
            print(f"f = {freq:2d} Hz: no bins above threshold")

    return {
        'amp_cpp': amp_cpp,
        'amp_py': amp_py,
        'phase_cpp': phase_cpp,
        'phase_py': phase_py,
        'rel_error_amp': rel_amp,
        'phase_diff': phase_sig if phase_sig.size else phase_diff,
        'amp_diff': amp_diff,
    }


def analyze_seismogram_statistics(original, recon_cpp, recon_py):
    """Analyzes statistics of amplitudes of reconstructed seismograms."""
    print("\n=== SEISMOGRAM STATISTICS ANALYSIS ===")

    signal_threshold = _significance_threshold(original)

    print("\n--- SEISMOGRAM AMPLITUDES ---")
    print(f"Mean  original: {np.mean(original):.6f}  |  C++: {np.mean(recon_cpp):.6f}  |  Python: {np.mean(recon_py):.6f}")
    print(f"Std   original: {np.std(original):.6f}  |  C++: {np.std(recon_cpp):.6f}  |  Python: {np.std(recon_py):.6f}")
    print(f"Max   original: {np.max(original):.6f}  |  C++: {np.max(recon_cpp):.6f}  |  Python: {np.max(recon_py):.6f}")
    print(f"Min   original: {np.min(original):.6f}  |  C++: {np.min(recon_cpp):.6f}  |  Python: {np.min(recon_py):.6f}")

    error_cpp = original - recon_cpp
    error_py = original - recon_py
    diff_cpp_py = recon_cpp - recon_py

    _print_abs_error_block("RECONSTRUCTION ERROR — C++ vs original", np.abs(error_cpp))
    _print_abs_error_block("RECONSTRUCTION ERROR — Python vs original", np.abs(error_py))
    _print_abs_error_block("RECONSTRUCTION DIFFERENCE — C++ vs Python", np.abs(diff_cpp_py))

    rel_cpp, mask_cpp = _masked_relative_error(error_cpp, original, signal_threshold)
    rel_py, mask_py = _masked_relative_error(error_py, original, signal_threshold)
    rel_cross, mask_cross = _masked_relative_error(diff_cpp_py, recon_cpp, signal_threshold)

    _print_rel_error_block("RELATIVE RECONSTRUCTION ERROR — C++", rel_cpp, mask_cpp, signal_threshold)
    _print_rel_error_block("RELATIVE RECONSTRUCTION ERROR — Python", rel_py, mask_py, signal_threshold)
    _print_rel_error_block("RELATIVE DIFFERENCE C++ vs Python", rel_cross, mask_cross, signal_threshold)

    return {
        'error_cpp': error_cpp,
        'error_py': error_py,
        'rel_error_cpp': rel_cpp,
        'rel_error_py': rel_py,
        'diff_cpp_py': diff_cpp_py,
        'rel_diff_cpp_py': rel_cross,
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
    
    # Absolute STFT amplitude differences
    axs[0, 2].hist(stft_stats['amp_diff'].flatten(), bins=50, alpha=0.7, density=True)
    axs[0, 2].set_title('|C++ - Python| STFT Amplitude Difference')
    axs[0, 2].set_xlabel('Absolute difference')
    axs[0, 2].set_ylabel('Density')
    axs[0, 2].set_yscale('log')

    # Reconstruction errors
    axs[1, 0].hist(seis_stats['error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[1, 0].hist(seis_stats['error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 0].set_title('Reconstruction Error Distribution')
    axs[1, 0].set_xlabel('Error')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].legend()
    
    # Relative reconstruction errors (significant samples only)
    if seis_stats['rel_error_cpp'].size:
        axs[1, 1].hist(seis_stats['rel_error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    if seis_stats['rel_error_py'].size:
        axs[1, 1].hist(seis_stats['rel_error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 1].set_title(f'Relative Reconstruction Error (|x| >= {SIGNIFICANCE_FRAC * 100:.0f}% peak)')
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