#ifndef STFT_HPP
#define STFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <kfr/dft.hpp>

// --- Types and enumerations ---
enum class BoundaryType { ZERO, EVEN };

// --- Helper functions ---
template<typename T>
std::vector<T> hann_window(int N) {
    std::vector<T> w(N);
    for (int i = 0; i < N; ++i) {
        w[i] = static_cast<T>(0.5) * (static_cast<T>(1.0) - std::cos(static_cast<T>(2.0 * M_PI) * i / (N - 1)));
    }
    return w;
}

template<typename T>
std::vector<T> apply_even_extension(const std::vector<T>& signal, int pad_size) {
    if (pad_size < 1) return signal;
    int n = static_cast<int>(signal.size());
    std::vector<T> extended(n + 2 * pad_size);
    for (int i = 0; i < n; ++i) extended[pad_size + i] = signal[i];
    for (int i = 0; i < pad_size; ++i) extended[i] = signal[pad_size - 1 - i];
    for (int i = 0; i < pad_size; ++i) extended[n + pad_size + i] = signal[n - 1 - i];
    return extended;
}

// --- KFR FFT engine (reusable plan + buffers) ---
template<typename T>
class KFR_FFT {
public:
    explicit KFR_FFT(int frame_size)
        : plan_(frame_size)
        , temp_(plan_.temp_size)
        , in_(frame_size)
        , freq_(plan_.complex_size())
    {}

    void r2c(const T* input) {
        for (size_t i = 0; i < in_.size(); ++i) in_[i] = input[i];
        plan_.execute(freq_, in_, temp_);
    }

    void c2r(const kfr::complex<T>* input) {
        for (size_t k = 0; k < freq_.size(); ++k) freq_[k] = input[k];
        plan_.execute(in_, freq_, temp_, kfr::cinvert_t{});
    }

    const kfr::complex<T>* freq_data() const { return freq_.data(); }
    const T* time_data() const { return in_.data(); }
    size_t n_freqs() const { return freq_.size(); }

private:
    kfr::dft_plan_real<T> plan_;
    kfr::univector<kfr::u8> temp_;
    kfr::univector<T> in_;
    kfr::univector<kfr::complex<T>> freq_;
};

// --- STFT processor (reuses FFT plan across frames/traces) ---
template<typename T>
class StftProcessor {
public:
    StftProcessor(int frame_size, int hop_size, BoundaryType boundary = BoundaryType::ZERO)
        : frame_size_(frame_size)
        , hop_size_(hop_size)
        , boundary_(boundary)
        , fft_(frame_size)
        , window_(hann_window<T>(frame_size))
        , window_sum_(std::accumulate(window_.begin(), window_.end(), static_cast<T>(0.0)))
    {
        if (frame_size <= 0 || hop_size <= 0) {
            throw std::invalid_argument("Invalid frame_size or hop_size");
        }
    }

    std::vector<std::vector<std::complex<T>>> forward(const std::vector<T>& signal) {
        std::vector<T> working_signal = signal;
        if (boundary_ == BoundaryType::EVEN) {
            working_signal = apply_even_extension(signal, frame_size_ / 2);
        }

        int n_frames = 1 + (static_cast<int>(working_signal.size()) - frame_size_ + hop_size_ - 1) / hop_size_;
        std::vector<std::vector<std::complex<T>>> stft_result(
            n_frames, std::vector<std::complex<T>>(fft_.n_freqs()));

        T scale = (window_sum_ > static_cast<T>(1e-9)) ? (static_cast<T>(1.0) / window_sum_) : static_cast<T>(1.0);
        std::vector<T> frame(frame_size_);

        for (int f = 0; f < n_frames; ++f) {
            int offset = f * hop_size_;
            for (int i = 0; i < frame_size_; ++i) {
                int idx = offset + i;
                frame[i] = (idx < static_cast<int>(working_signal.size()))
                    ? working_signal[idx] * window_[i]
                    : static_cast<T>(0.0);
            }
            fft_.r2c(frame.data());
            for (size_t k = 0; k < fft_.n_freqs(); ++k) {
                const auto& bin = fft_.freq_data()[k];
                stft_result[f][k] = std::complex<T>(bin.real() * scale, bin.imag() * scale);
            }
        }
        return stft_result;
    }

    std::vector<T> inverse(const std::vector<std::vector<std::complex<T>>>& stft_result, int original_length) {
        if (stft_result.empty()) return {};

        int n_frames = static_cast<int>(stft_result.size());
        int n_freqs = static_cast<int>(fft_.n_freqs());
        int signal_length = (n_frames - 1) * hop_size_ + frame_size_;

        std::vector<T> output(signal_length, static_cast<T>(0.0));
        std::vector<T> window_sums(signal_length, static_cast<T>(0.0));

        T inv_scale = (window_sum_ > static_cast<T>(1e-9)) ? window_sum_ : static_cast<T>(1.0);
        std::vector<kfr::complex<T>> freq(n_freqs);

        for (int f = 0; f < n_frames; ++f) {
            for (int k = 0; k < n_freqs; ++k) {
                freq[k] = kfr::complex<T>(
                    stft_result[f][k].real() * inv_scale,
                    stft_result[f][k].imag() * inv_scale);
            }
            fft_.c2r(freq.data());
            int offset = f * hop_size_;
            for (int i = 0; i < frame_size_; ++i) {
                if (offset + i < signal_length) {
                    output[offset + i] += (fft_.time_data()[i] / static_cast<T>(frame_size_)) * window_[i];
                    window_sums[offset + i] += window_[i] * window_[i];
                }
            }
        }

        for (size_t i = 0; i < output.size(); ++i) {
            if (window_sums[i] > static_cast<T>(1e-9)) {
                output[i] /= window_sums[i];
            }
        }

        if (original_length <= 0) return output;
        int pad_size = (boundary_ == BoundaryType::EVEN) ? frame_size_ / 2 : 0;
        if (output.size() >= static_cast<size_t>(original_length + pad_size)) {
            std::vector<T> final_signal(original_length);
            std::copy(output.begin() + pad_size, output.begin() + pad_size + original_length, final_signal.begin());
            return final_signal;
        }
        if (original_length < static_cast<int>(output.size())) output.resize(original_length);
        return output;
    }

private:
    int frame_size_;
    int hop_size_;
    BoundaryType boundary_;
    KFR_FFT<T> fft_;
    std::vector<T> window_;
    T window_sum_;
};

// --- Backward-compatible free functions ---
template<typename T>
std::vector<std::vector<std::complex<T>>> STFT_forward(
    const std::vector<T>& signal, int frame_size, int hop_size, BoundaryType boundary = BoundaryType::ZERO
) {
    StftProcessor<T> proc(frame_size, hop_size, boundary);
    return proc.forward(signal);
}

template<typename T>
std::vector<T> STFT_inverse(
    const std::vector<std::vector<std::complex<T>>>& stft_result,
    int frame_size, int hop_size, int original_length, BoundaryType boundary
) {
    StftProcessor<T> proc(frame_size, hop_size, boundary);
    return proc.inverse(stft_result, original_length);
}

#endif // STFT_HPP
