#include <cmath>
#include <fftw3.h>
#include <stdexcept>
#include <stft.hpp>

using std::vector;
using std::complex;

using Complex = std::complex<double>;

// Generate Hann window function
vector<double> hann_window(int N) {
    vector<double> w(N);
    for (int i = 0; i < N; ++i) {
        w[i] = 0.5 * (1 - cos(2 * M_PI * i / N));
    }
    return w;
}

// STFT: analog of scipy.signal.stft (single channel)
vector<vector<Complex>> STFT_forward(const vector<double>& signal, int frame_size, int hop_size) {
    if (frame_size <= 0 || hop_size <= 0) throw std::invalid_argument("frame_size and hop_size must be > 0");

    // Add padding as in scipy.signal.stft with boundary='zeros' and padded=True
    // scipy adds padding to get an integer number of frames
    int n_frames = 1 + (signal.size() - frame_size + hop_size - 1) / hop_size;
    int padded_size = (n_frames - 1) * hop_size + frame_size;
    vector<vector<Complex>> stft_result(n_frames, vector<Complex>(frame_size / 2 + 1));

    vector<double> window = hann_window(frame_size);

    fftw_plan plan;
    double* in = fftw_alloc_real(frame_size);
    fftw_complex* out = fftw_alloc_complex(frame_size / 2 + 1);
    plan = fftw_plan_dft_r2c_1d(frame_size, in, out, FFTW_MEASURE);

    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            int signal_idx = offset + i;
            if (signal_idx < signal.size()) {
                in[i] = signal[signal_idx] * window[i];
            } else {
                in[i] = 0.0;  // zero-padding
            }
        }

        fftw_execute(plan);
        for (int k = 0; k < frame_size / 2 + 1; ++k) {
            stft_result[f][k] = Complex(out[k][0], out[k][1]);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return stft_result;
}

// Inverse STFT: analog of scipy.signal.istft
vector<double> STFT_inverse(const vector<vector<Complex>>& stft_result, int frame_size, int hop_size, int original_length) {
    int n_frames = stft_result.size();
    int signal_length = (n_frames - 1) * hop_size + frame_size;
    vector<double> output(signal_length, 0.0);
    vector<double> window = hann_window(frame_size);
    vector<double> norm(signal_length, 0.0);

    fftw_plan plan;
    fftw_complex* in = fftw_alloc_complex(frame_size / 2 + 1);
    double* out = fftw_alloc_real(frame_size);
    plan = fftw_plan_dft_c2r_1d(frame_size, in, out, FFTW_MEASURE);

    for (int f = 0; f < n_frames; ++f) {
        for (int k = 0; k < frame_size / 2 + 1; ++k) {
            in[k][0] = stft_result[f][k].real();
            in[k][1] = stft_result[f][k].imag();
        }

        fftw_execute(plan);

        for (int i = 0; i < frame_size; ++i) {
            double w = window[i];
            int idx = f * hop_size + i;
            if (idx < signal_length) {
                // FFTW returns result multiplied by frame_size
                output[idx] += (out[i] / frame_size) * w;
                norm[idx] += w * w;
            }
        }
    }

    // Normalization as in scipy
    for (int i = 0; i < signal_length; ++i) {
        if (norm[i] > 1e-8) {
            output[i] /= norm[i];
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // Обрезаем до оригинального размера, как в Python версии
    if (original_length > 0 && original_length < output.size()) {
        output.resize(original_length);
    }

    return output;
}

std::vector<float> hann_window_float(int N) {
    vector<float> w(N);
    for (int i = 0; i < N; ++i)
        w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / N));
    return w;
}

std::vector<std::vector<ComplexF>> STFT_forward_float(const std::vector<float>& signal, int frame_size, int hop_size) {
    // Add padding as in scipy.signal.stft with boundary='zeros' and padded=True
    // scipy adds padding to get an integer number of frames
    int n_frames = 1 + (signal.size() - frame_size + hop_size - 1) / hop_size;
    int padded_size = (n_frames - 1) * hop_size + frame_size;
    vector<vector<ComplexF>> stft_result(n_frames, vector<ComplexF>(frame_size / 2 + 1));

    vector<float> window = hann_window_float(frame_size);

    float* in = fftwf_alloc_real(frame_size);
    fftwf_complex* out = fftwf_alloc_complex(frame_size / 2 + 1);
    auto plan = fftwf_plan_dft_r2c_1d(frame_size, in, out, FFTW_MEASURE);

    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            int signal_idx = offset + i;
            if (signal_idx < signal.size()) {
                in[i] = signal[signal_idx] * window[i];
            } else {
                in[i] = 0.0f;  // zero-padding
            }
        }

        fftwf_execute(plan);
        for (int k = 0; k < frame_size / 2 + 1; ++k)
            stft_result[f][k] = ComplexF(out[k][0], out[k][1]);
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
    return stft_result;
}

std::vector<float> STFT_inverse_float(
    const std::vector<std::vector<ComplexF>>& stft_result,
    int frame_size,
    int hop_size,
    int original_length
) {
    int n_frames = stft_result.size();
    int signal_length = (n_frames - 1) * hop_size + frame_size;
    std::vector<float> output(signal_length, 0.0f);
    std::vector<float> window = hann_window_float(frame_size);
    std::vector<float> norm(signal_length, 0.0f);

    fftwf_complex* in = fftwf_alloc_complex(frame_size / 2 + 1);
    float* out = fftwf_alloc_real(frame_size);
    auto plan = fftwf_plan_dft_c2r_1d(frame_size, in, out, FFTW_MEASURE);

    for (int f = 0; f < n_frames; ++f) {
        for (int k = 0; k < frame_size / 2 + 1; ++k) {
            in[k][0] = stft_result[f][k].real();
            in[k][1] = stft_result[f][k].imag();
        }

        fftwf_execute(plan);

        for (int i = 0; i < frame_size; ++i) {
            float w = window[i];
            int idx = f * hop_size + i;
            if (idx < signal_length) {
                // FFTW returns result multiplied by frame_size
                output[idx] += (out[i] / frame_size) * w;
                norm[idx] += w * w;
            }
        }
    }

    // Normalization as in scipy
    for (int i = 0; i < signal_length; ++i) {
        if (norm[i] > 1e-8f) {
            output[i] /= norm[i];
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    // Trim to original size, as in Python version
    if (original_length > 0 && original_length < output.size()) {
        output.resize(original_length);
    }

    return output;
}


