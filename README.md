# STFT/ISTFT Implementation Comparison

This project compares C++ and Python implementations of Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT) with identical parameters to ensure mathematical equivalence.

## Files

- `cpp_stft.cpp` - C++ implementation using FFTW library
- `py_stft.py` - Python implementation using SciPy
- `stft.hpp` / `stft.cpp` - C++ STFT/ISTFT functions
- `compare_results.py` - Script to compare C++ and Python results
- `compile.sh` - Compilation script for C++ code
- `data/input.bin` - Input seismic data (220 traces × 501 samples)

## Parameters

Both implementations use identical parameters:
- **Frame size**: 64 samples
- **Hop size**: 32 samples  
- **Window**: Hann window
- **Boundary**: Even-symmetric padding
- **Sampling rate**: 250 Hz (dt = 0.004s)

## Usage

### 1. Compile and run C++ version
```bash
./compile.sh
./cpp_stft
```

### 2. Run Python version
```bash
python py_stft.py
```

### 3. Compare results
```bash
python compare_results.py
```

## Output Files

- `data/stft_cpp.bin` - C++ STFT frames (complex data)
- `data/stft_py.bin` - Python STFT frames (complex data)
- `data/output_cpp.bin` - C++ reconstructed signal
- `data/output_py.bin` - Python reconstructed signal

## Analysis

Both programs provide comprehensive analysis:

### STFT Spectrum Analysis
- Magnitude statistics (min, max, mean)
- Phase statistics (min, max, mean)
- Total number of samples

### Reconstructed Signal Analysis  
- Signal statistics (min, max, mean)
- Total number of samples

### Reconstruction Quality Analysis
- Max/mean/RMS differences between original and reconstructed
- Relative error percentage

## Expected Results

With properly aligned implementations:
- **STFT magnitudes**: Should match to machine precision
- **STFT phases**: May have small differences due to numerical precision
- **Reconstructed signals**: Should match to machine precision
- **Reconstruction quality**: Should show minimal error (< 1e-10)

## Dependencies

### C++
- FFTW3 library
- C++11 or later

### Python  
- NumPy
- SciPy
- Matplotlib (optional, for plotting)

## Notes

- Both implementations use `boundary='even'` for consistent padding
- C++ uses FFTW for FFT calculations
- Python uses SciPy's optimized STFT/ISTFT functions
- Binary file formats are compatible between C++ and Python



