#include "stft.hpp" // Предполагается, что здесь BoundaryType и т.д.
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>

#include <type_traits>

// Основной шаблон структуры (неопределенный)
template<typename T>
struct FFTW_Traits;

// Специализация для double
template<>
struct FFTW_Traits<double> {
    using real_type = double;
    using complex_type = fftw_complex;
    using plan_type = fftw_plan;

    static real_type* alloc_real(size_t n) { return fftw_alloc_real(n); }
    static complex_type* alloc_complex(size_t n) { return fftw_alloc_complex(n); }
    static void free(void* p) { fftw_free(p); }
    static void execute(plan_type p) { fftw_execute(p); }
    static void destroy_plan(plan_type p) { fftw_destroy_plan(p); }

    static plan_type plan_dft_r2c_1d(int n, real_type* in, complex_type* out, unsigned flags) {
        return fftw_plan_dft_r2c_1d(n, in, out, flags);
    }
    static plan_type plan_dft_c2r_1d(int n, complex_type* in, real_type* out, unsigned flags) {
        return fftw_plan_dft_c2r_1d(n, in, out, flags);
    }
};

// Специализация для float
template<>
struct FFTW_Traits<float> {
    using real_type = float;
    using complex_type = fftwf_complex;
    using plan_type = fftwf_plan;

    static real_type* alloc_real(size_t n) { return fftwf_alloc_real(n); }
    static complex_type* alloc_complex(size_t n) { return fftwf_alloc_complex(n); }
    static void free(void* p) { fftwf_free(p); }
    static void execute(plan_type p) { fftwf_execute(p); }
    static void destroy_plan(plan_type p) { fftwf_destroy_plan(p); }

    static plan_type plan_dft_r2c_1d(int n, real_type* in, complex_type* out, unsigned flags) {
        return fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }
    static plan_type plan_dft_c2r_1d(int n, complex_type* in, real_type* out, unsigned flags) {
        return fftwf_plan_dft_c2r_1d(n, in, out, flags);
    }
};

using std::vector;
using std::complex;

// Шаблонная функция для окна Ханна
template<typename T>
vector<T> hann_window(int N) {
    vector<T> w(N);
    for (int i = 0; i < N; ++i) {
        // Используем std::cos, который перегружен для float и double
        w[i] = static_cast<T>(0.5) * (static_cast<T>(1.0) - std::cos(static_cast<T>(2.0 * M_PI) * i / (N - 1)));
    }
    return w;
}

// Шаблонная функция для расширения сигнала
template<typename T>
vector<T> apply_even_extension(const vector<T>& signal, int pad_size) {
    if (pad_size < 1) return signal;

    int n = signal.size();
    vector<T> extended(n + 2 * pad_size);

    for (int i = 0; i < n; ++i) extended[pad_size + i] = signal[i];
    for (int i = 0; i < pad_size; ++i) extended[i] = signal[pad_size - 1 - i];
    for (int i = 0; i < pad_size; ++i) extended[n + pad_size + i] = signal[n - 1 - i];
    
    return extended;
}

// Шаблонная функция STFT_forward
template<typename T>
vector<vector<complex<T>>> STFT_forward(const vector<T>& signal, int frame_size, int hop_size, BoundaryType boundary) {
    if (frame_size <= 0 || hop_size <= 0) throw std::invalid_argument("Invalid frame_size or hop_size");
    
    // Получаем типы и функции из нашей структуры Traits
    using Traits = FFTW_Traits<T>;

    vector<T> working_signal = signal;
    if (boundary == BoundaryType::EVEN) {
        int pad_size = frame_size / 2;
        working_signal = apply_even_extension(signal, pad_size);
    }

    int n_frames = 1 + (working_signal.size() - frame_size + hop_size - 1) / hop_size;
    
    vector<vector<complex<T>>> stft_result(n_frames, vector<complex<T>>(frame_size / 2 + 1));
    vector<T> window = hann_window<T>(frame_size);

    typename Traits::real_type* in = Traits::alloc_real(frame_size);
    typename Traits::complex_type* out = Traits::alloc_complex(frame_size / 2 + 1);
    typename Traits::plan_type plan = Traits::plan_dft_r2c_1d(frame_size, in, out, FFTW_MEASURE);

    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            int idx = offset + i;
            in[i] = (idx < working_signal.size()) ? working_signal[idx] * window[i] : static_cast<T>(0.0);
        }

        Traits::execute(plan);
        for (int k = 0; k < frame_size / 2 + 1; ++k) {
            stft_result[f][k] = complex<T>(out[k][0], out[k][1]);
        }
    }

    Traits::destroy_plan(plan);
    Traits::free(in);
    Traits::free(out);
    return stft_result;
}

// Шаблонная функция STFT_inverse
template<typename T>
vector<T> STFT_inverse(const vector<vector<complex<T>>>& stft_result, int frame_size, int hop_size, int original_length) {
    if (stft_result.empty()) return {};

    using Traits = FFTW_Traits<T>;
    
    int n_frames = stft_result.size();
    int n_freqs = frame_size / 2 + 1;
    int signal_length = (n_frames - 1) * hop_size + frame_size;

    vector<T> output(signal_length, static_cast<T>(0.0));
    vector<T> window_sums(signal_length, static_cast<T>(0.0));
    vector<T> window = hann_window<T>(frame_size);

    typename Traits::complex_type* in = Traits::alloc_complex(n_freqs);
    typename Traits::real_type* out = Traits::alloc_real(frame_size);
    typename Traits::plan_type plan = Traits::plan_dft_c2r_1d(frame_size, in, out, FFTW_ESTIMATE);

    for (int f = 0; f < n_frames; ++f) {
        for (int k = 0; k < n_freqs; ++k) {
            in[k][0] = stft_result[f][k].real();
            in[k][1] = stft_result[f][k].imag();
        }

        Traits::execute(plan);

        int offset = f * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            if (offset + i < signal_length) {
                output[offset + i] += (out[i] / frame_size) * window[i];
                window_sums[offset + i] += window[i] * window[i];
            }
        }
    }

    Traits::destroy_plan(plan);
    Traits::free(in);
    Traits::free(out);

    for (size_t i = 0; i < output.size(); ++i) {
        if (window_sums[i] > static_cast<T>(1e-9)) {
            output[i] /= window_sums[i];
        }
    }
    
    int pad_size = frame_size / 2;
    if (original_length > 0 && output.size() >= original_length + pad_size) {
        vector<T> final_signal(original_length);
        for(int i = 0; i < original_length; ++i) {
            final_signal[i] = output[pad_size + i];
        }
        return final_signal;
    } else {
        if (original_length > 0 && original_length < output.size()) {
            output.resize(original_length);
        }
        return output;
    }
}