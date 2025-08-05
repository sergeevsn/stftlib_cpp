#ifndef STFT_HPP
#define STFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <fftw3.h>

// --- Типы и перечисления ---

/**
 * @brief Типы граничного дополнения сигнала.
 */
enum class BoundaryType {
    ZERO,   // Дополнение нулями (аналог 'zeros' в scipy)
    EVEN    // Симметричное "четное" дополнение (аналог 'reflect' в numpy или 'even' в scipy)
};


// --- Вспомогательная структура для работы с FFTW ---

/**
 * @brief Шаблонная структура (Type Traits) для абстрагирования от float/double версий FFTW.
 *        Предоставляет правильные типы и указатели на функции FFTW в зависимости от типа T.
 */
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


// --- Вспомогательные шаблонные функции (реализация в .hpp) ---

/**
 * @brief Создает окно Ханна.
 * @tparam T Тип данных (float или double).
 * @param N Размер окна.
 * @return Вектор с коэффициентами окна.
 */
template<typename T>
std::vector<T> hann_window(int N) {
    std::vector<T> w(N);
    for (int i = 0; i < N; ++i) {
        w[i] = static_cast<T>(0.5) * (static_cast<T>(1.0) - std::cos(static_cast<T>(2.0 * M_PI) * i / (N - 1)));
    }
    return w;
}

/**
 * @brief Применяет "четное" симметричное дополнение к сигналу.
 */
template<typename T>
std::vector<T> apply_even_extension(const std::vector<T>& signal, int pad_size) {
    if (pad_size < 1) return signal;

    int n = signal.size();
    std::vector<T> extended(n + 2 * pad_size);

    for (int i = 0; i < n; ++i) extended[pad_size + i] = signal[i];
    for (int i = 0; i < pad_size; ++i) extended[i] = signal[pad_size - 1 - i];
    for (int i = 0; i < pad_size; ++i) extended[n + pad_size + i] = signal[n - 1 - i];
    
    return extended;
}


// --- Основные функции STFT/iSTFT ---

/**
 * @brief Прямое кратковременное преобразование Фурье (STFT).
 * @tparam T Тип данных сигнала (float или double).
 * @param signal Входной сигнал.
 * @param frame_size Размер окна (кадра).
 * @param hop_size Шаг (смещение) окна.
 * @param boundary Тип дополнения границ сигнала.
 * @return 2D вектор комплексных чисел (спектрограмма).
 */
template<typename T>
std::vector<std::vector<std::complex<T>>> STFT_forward(
    const std::vector<T>& signal,
    int frame_size,
    int hop_size,
    BoundaryType boundary = BoundaryType::ZERO
) {
    if (frame_size <= 0 || hop_size <= 0) throw std::invalid_argument("Invalid frame_size or hop_size");
    
    using Traits = FFTW_Traits<T>;
    std::vector<T> working_signal = signal;

    if (boundary == BoundaryType::EVEN) {
        int pad_size = frame_size / 2;
        working_signal = apply_even_extension(signal, pad_size);
    }

    int n_frames = 1 + (working_signal.size() - frame_size + hop_size - 1) / hop_size;
    
    std::vector<std::vector<std::complex<T>>> stft_result(n_frames, std::vector<std::complex<T>>(frame_size / 2 + 1));
    std::vector<T> window = hann_window<T>(frame_size);

    typename Traits::real_type* in = Traits::alloc_real(frame_size);
    typename Traits::complex_type* out = Traits::alloc_complex(frame_size / 2 + 1);
    typename Traits::plan_type plan = Traits::plan_dft_r2c_1d(frame_size, in, out, FFTW_ESTIMATE);

    for (int f = 0; f < n_frames; ++f) {
        int offset = f * hop_size;
        for (int i = 0; i < frame_size; ++i) {
            int idx = offset + i;
            in[i] = (idx < working_signal.size()) ? working_signal[idx] * window[i] : static_cast<T>(0.0);
        }

        Traits::execute(plan);
        for (int k = 0; k < frame_size / 2 + 1; ++k) {
            stft_result[f][k] = std::complex<T>(out[k][0], out[k][1]);
        }
    }

    Traits::destroy_plan(plan);
    Traits::free(in);
    Traits::free(out);
    return stft_result;
}

/**
 * @brief Обратное кратковременное преобразование Фурье (iSTFT).
 * @tparam T Тип данных сигнала (float или double).
 * @param stft_result Входная спектрограмма.
 * @param frame_size Размер окна, использованный в STFT.
 * @param hop_size Шаг окна, использованный в STFT.
 * @param original_length Исходная длина сигнала для корректной обрезки.
 * @param boundary Тип дополнения, который ИСПОЛЬЗОВАЛСЯ в прямом STFT.
 * @return Восстановленный сигнал.
 */
 template<typename T>
 std::vector<T> STFT_inverse(
     const std::vector<std::vector<std::complex<T>>>& stft_result,
     int frame_size,
     int hop_size,
     int original_length,
     BoundaryType boundary // <--- ИЗМЕНЕНИЕ ЗДЕСЬ: добавлен параметр
 ) {
     if (stft_result.empty()) return {};
 
     using Traits = FFTW_Traits<T>;
     
     int n_frames = stft_result.size();
     int n_freqs = frame_size / 2 + 1;
     int signal_length = (n_frames - 1) * hop_size + frame_size;
 
     std::vector<T> output(signal_length, static_cast<T>(0.0));
     std::vector<T> window_sums(signal_length, static_cast<T>(0.0));
     std::vector<T> window = hann_window<T>(frame_size);
 
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
     
     if (original_length <= 0) {
         return output;
     }
 
     // Теперь эта строка будет работать, так как 'boundary' определен
     int pad_size = (boundary == BoundaryType::EVEN) ? frame_size / 2 : 0;
     
     if (output.size() >= original_length + pad_size) {
         // Создаем вектор нужной длины и копируем срез
         std::vector<T> final_signal(original_length);
         if (pad_size > 0) {
              std::copy(output.begin() + pad_size, output.begin() + pad_size + original_length, final_signal.begin());
         } else {
              std::copy(output.begin(), output.begin() + original_length, final_signal.begin());
         }
         return final_signal;
     } else {
         // Резервный вариант, если что-то пошло не так
         if (original_length < output.size()) {
             output.resize(original_length);
         }
         return output;
     }
 }
 
 #endif // STFT_HPP