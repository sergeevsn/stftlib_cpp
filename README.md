# STFT Library

A C++ implementation of Short-Time Fourier Transform (STFT) and Inverse STFT with Python comparison utilities.

## Overview

This library provides efficient C++ implementations of STFT and ISTFT algorithms using FFTW3, with full compatibility to scipy.signal.stft/istft behavior. The implementation supports both single-precision (float) and double-precision (double) data types.

## Features

- **STFT Forward Transform**: Converts time-domain signals to time-frequency representation
- **STFT Inverse Transform**: Reconstructs time-domain signals from time-frequency data
- **Hann Window**: Built-in Hann window function for spectral analysis
- **Padding Support**: Automatic zero-padding for boundary handling
- **FFTW3 Integration**: High-performance FFT using FFTW3 library
- **Python Comparison**: Utilities to compare C++ results with scipy.signal

## Files Description

### Core Library Files

- **`stft.hpp`** - Header file containing function declarations and type definitions
- **`stft.cpp`** - Implementation of STFT/ISTFT algorithms for both float and double precision

### Main Applications

- **`cpp_stft.cpp`** - Main C++ application that processes seismic data using STFT/ISTFT
  - Reads binary seismic data (220 traces × 501 samples)
  - Applies STFT to each trace with configurable parameters
  - Performs inverse STFT reconstruction
  - Saves reconstructed data to binary file
  - Outputs time and frequency axis information

- **`py_stft.py`** - Python reference implementation using scipy.signal
  - Demonstrates equivalent functionality using scipy.signal.stft/istft
  - Serves as a reference for correct behavior
  - Used for validation and comparison with C++ implementation

- **`compare.py`** - Visualization and comparison utility
  - Loads original, C++ reconstructed, and Python reconstructed data
  - Creates side-by-side comparison plots
  - Uses common color scale for fair comparison
  - Helps identify any discrepancies between implementations

### Build and Utility Files

- **`compile.sh`** - Compilation script for C++ implementation
- **`data/`** - Directory containing input/output binary files

## Usage

### Compilation

```bash
./compile.sh
```

### Running C++ Implementation

```bash
./cpp_stft
```

### Running Python Implementation

```bash
python py_stft.py
```

### Comparing Results

```bash
python compare.py
```

## Parameters

The library uses the following default parameters:
- **Frame Size**: 64 samples
- **Hop Size**: 32 samples (50% overlap)
- **Window**: Hann window
- **Padding**: Zero-padding with boundary='zeros' and padded=True
- **Data**: 220 traces × 501 samples each

## Algorithm Details

### STFT Forward Transform
1. Applies Hann window to signal frames
2. Performs FFT on each frame
3. Returns one-sided spectrum (positive frequencies only)
4. Handles zero-padding for boundary conditions

### STFT Inverse Transform
1. Performs inverse FFT on each time-frequency frame
2. Applies Hann window to reconstructed frames
3. Overlap-add reconstruction with proper normalization
4. Trims result to original signal length

### Normalization
The implementation correctly handles FFTW scaling and window overlap normalization to ensure perfect reconstruction (within numerical precision).

## Dependencies

- **C++**: FFTW3 library
- **Python**: numpy, scipy, matplotlib

## Performance

The C++ implementation provides significant performance improvements over Python/scipy for large datasets while maintaining identical numerical accuracy.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 