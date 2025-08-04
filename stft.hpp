#ifndef STFT_HPP
#define STFT_HPP

#include <vector>
#include <complex>

// Types for convenience
using Complex = std::complex<double>;
using ComplexF = std::complex<float>;

// Hann windows
std::vector<double> hann_window(int N);
std::vector<float> hann_window_float(int N);

// STFT (double)
std::vector<std::vector<Complex>> STFT_forward(
    const std::vector<double>& signal,
    int frame_size,
    int hop_size
);

std::vector<double> STFT_inverse(
    const std::vector<std::vector<Complex>>& stft_result,
    int frame_size,
    int hop_size,
    int original_length = 0
);

// STFT (float)
std::vector<std::vector<ComplexF>> STFT_forward_float(
    const std::vector<float>& signal,
    int frame_size,
    int hop_size
);

std::vector<float> STFT_inverse_float(
    const std::vector<std::vector<std::complex<float>>>& stft_result,
    int frame_size,
    int hop_size,
    int original_length
);

#endif // STFT_HPP
