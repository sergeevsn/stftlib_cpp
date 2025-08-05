import numpy as np
import matplotlib.pyplot as plt
import os

# === 1. Параметры и пути к файлам ===
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


# === 2. Функции для загрузки данных ===

def load_stft_bin_cpp(path):
    """Загружает C++ STFT данные (traces, frames, freqs)."""
    with open(path, 'rb') as f:
        n_traces_f, n_frames_f, n_freqs_f = np.frombuffer(f.read(12), dtype=np.int32)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_traces_f, n_frames_f, n_freqs_f, 2)
        return data[..., 0] + 1j * data[..., 1]

def load_stft_bin_py(path):
    """
    *** ИСПРАВЛЕННАЯ И ОКОНЧАТЕЛЬНАЯ ВЕРСИЯ ***
    Корректно загружает Python STFT данные, которые были сохранены
    в формате (traces, frames, freqs).
    """
    with open(path, 'rb') as f:
        # 1. Читаем заголовок. Получаем: 220, 17, 33
        n_traces_f, n_frames_f, n_freqs_f = np.frombuffer(f.read(12), dtype=np.int32)
        print(n_traces_f, n_frames_f, n_freqs_f)
        
        # 2. Читаем данные и восстанавливаем комплексные числа
        raw_data = np.frombuffer(f.read(), dtype=np.float32)
        complex_data = raw_data[::2] + 1j * raw_data[1::2]
        
        # 3. Придаем массиву правильную форму (220, 17, 33).
        #    Никакого транспонирования не нужно.
        return complex_data.reshape(n_traces_f, n_frames_f, n_freqs_f)

def load_seismogram_bin(path, traces, samples):
    """Загружает данные сейсмограммы."""
    return np.fromfile(path, dtype=np.float32).reshape(traces, samples)


# === 3. Функции для визуализации (без изменений) ===

def plot_seismogram_comparison(original, recon_cpp, recon_py, output_path):
    print("Создание графика сравнения сейсмограмм...")
    diff_cpp = original - recon_cpp
    diff_py = original - recon_py
    vmin, vmax = np.percentile(original, [2, 98])
   
    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
    fig.suptitle('Сравнение исходной и восстановленных сейсмограмм', fontsize=16)
    axs[0, 0].imshow(original.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 0].set_title('Исходная')
    axs[0, 0].set_ylabel('Отсчеты времени')
    axs[0, 1].imshow(recon_cpp.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 1].set_title('Восстановлено (C++)')
    axs[0, 2].imshow(diff_cpp.T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')
    axs[0, 2].set_title('Разница (Исходный - C++)')
    axs[1, 0].imshow(original.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 0].set_ylabel('Отсчеты времени')
    axs[1, 0].set_xlabel('Номер трассы')
    axs[1, 1].imshow(recon_py.T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 1].set_title('Восстановлено (Python)')
    axs[1, 1].set_xlabel('Номер трассы')
    axs[1, 2].imshow(diff_py.T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='upper')
    axs[1, 2].set_title('Разница (Исходный - Python)')
    axs[1, 2].set_xlabel('Номер трассы')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  График сохранен в {output_path}")
    plt.close(fig)

def plot_stft_comparison(stft_cpp, stft_py, f_axis, t_axis, output_path):
    print("Создание графика сравнения STFT...")
    freqs_to_plot_hz = [20, 40, 60]
    freq_indices = [np.argmin(np.abs(f_axis - f)) for f in freqs_to_plot_hz]
    fig, axs = plt.subplots(len(freqs_to_plot_hz), 2, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle('Сравнение амплитудных спектров STFT', fontsize=16)
    for i, freq_idx in enumerate(freq_indices):
        freq_val = f_axis[freq_idx]
        cpp_slice = np.abs(stft_cpp[:, :, freq_idx])
        py_slice = np.abs(stft_py[:, :, freq_idx])        
        axs[i, 0].imshow(cpp_slice.T, aspect='auto', cmap='viridis', origin='upper', extent=[0, N_TRACES, t_axis[0], t_axis[-1]])
        axs[i, 0].set_ylabel(f'f = {freq_val:.1f} Hz\nВремя (с)')
        im = axs[i, 1].imshow(py_slice.T, aspect='auto', cmap='viridis', origin='upper', extent=[0, N_TRACES, t_axis[0], t_axis[-1]])
       
    axs[0, 0].set_title('C++ STFT')
    axs[0, 1].set_title('Python STFT')
    axs[-1, 0].set_xlabel('Номер трассы')
    axs[-1, 1].set_xlabel('Номер трассы')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  График сохранен в {output_path}")
    plt.close(fig)


# === 3.5. Функции для статистического анализа ===

def analyze_stft_statistics(stft_cpp, stft_py, f_axis, t_axis):
    """Анализирует статистику амплитудных и фазовых спектров STFT."""
    print("\n=== АНАЛИЗ СТАТИСТИКИ STFT ===")
    
    # Вычисляем амплитудные и фазовые спектры
    amp_cpp = np.abs(stft_cpp)
    amp_py = np.abs(stft_py)
    phase_cpp = np.angle(stft_cpp)
    phase_py = np.angle(stft_py)
    
    # Статистика по амплитудным спектрам
    print("\n--- АМПЛИТУДНЫЕ СПЕКТРЫ ---")
    print(f"Среднее значение C++: {np.mean(amp_cpp):.6f}")
    print(f"Среднее значение Python: {np.mean(amp_py):.6f}")
    print(f"Разность средних: {np.mean(amp_cpp) - np.mean(amp_py):.2e}")
    
    print(f"Стандартное отклонение C++: {np.std(amp_cpp):.6f}")
    print(f"Стандартное отклонение Python: {np.std(amp_py):.6f}")
    print(f"Разность СКО: {np.std(amp_cpp) - np.std(amp_py):.2e}")
    
    print(f"Максимум C++: {np.max(amp_cpp):.6f}")
    print(f"Максимум Python: {np.max(amp_py):.6f}")
    print(f"Разность максимумов: {np.max(amp_cpp) - np.max(amp_py):.2e}")
    
    print(f"Минимум C++: {np.min(amp_cpp):.6f}")
    print(f"Минимум Python: {np.min(amp_py):.6f}")
    print(f"Разность минимумов: {np.min(amp_cpp) - np.min(amp_py):.2e}")
    
    # Относительная ошибка амплитуд
    rel_error_amp = np.abs(amp_cpp - amp_py) / (np.abs(amp_cpp) + 1e-10)
    print(f"Средняя относительная ошибка амплитуд: {np.mean(rel_error_amp):.2e}")
    print(f"Максимальная относительная ошибка амплитуд: {np.max(rel_error_amp):.2e}")
    
    # Статистика по фазовым спектрам
    print("\n--- ФАЗОВЫЕ СПЕКТРЫ ---")
    print(f"Среднее значение C++: {np.mean(phase_cpp):.6f}")
    print(f"Среднее значение Python: {np.mean(phase_py):.6f}")
    print(f"Разность средних: {np.mean(phase_cpp) - np.mean(phase_py):.2e}")
    
    print(f"Стандартное отклонение C++: {np.std(phase_cpp):.6f}")
    print(f"Стандартное отклонение Python: {np.std(phase_py):.6f}")
    print(f"Разность СКО: {np.std(phase_cpp) - np.std(phase_py):.2e}")
    
    # Абсолютная ошибка фаз (с учетом циклического характера)
    phase_diff = np.abs(phase_cpp - phase_py)
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)  # Учитываем циклический характер
    print(f"Средняя абсолютная ошибка фаз: {np.mean(phase_diff):.2e}")
    print(f"Максимальная абсолютная ошибка фаз: {np.max(phase_diff):.2e}")
    
    # Статистика по частотам
    print("\n--- СТАТИСТИКА ПО ЧАСТОТАМ ---")
    for i, freq in enumerate([20, 40, 60]):
        freq_idx = np.argmin(np.abs(f_axis - freq))
        amp_cpp_freq = amp_cpp[:, :, freq_idx]
        amp_py_freq = amp_py[:, :, freq_idx]
        
        rel_error_freq = np.abs(amp_cpp_freq - amp_py_freq) / (np.abs(amp_cpp_freq) + 1e-10)
        print(f"Частота {freq} Hz - средняя относительная ошибка: {np.mean(rel_error_freq):.2e}")
    
    return {
        'amp_cpp': amp_cpp,
        'amp_py': amp_py,
        'phase_cpp': phase_cpp,
        'phase_py': phase_py,
        'rel_error_amp': rel_error_amp,
        'phase_diff': phase_diff
    }

def analyze_seismogram_statistics(original, recon_cpp, recon_py):
    """Анализирует статистику амплитуд восстановленных сейсмограмм."""
    print("\n=== АНАЛИЗ СТАТИСТИКИ СЕЙСМОГРАММ ===")
    
    # Статистика по амплитудам
    print("\n--- АМПЛИТУДЫ СЕЙСМОГРАММ ---")
    print(f"Среднее значение исходной: {np.mean(original):.6f}")
    print(f"Среднее значение C++: {np.mean(recon_cpp):.6f}")
    print(f"Среднее значение Python: {np.mean(recon_py):.6f}")
    
    print(f"Стандартное отклонение исходной: {np.std(original):.6f}")
    print(f"Стандартное отклонение C++: {np.std(recon_cpp):.6f}")
    print(f"Стандартное отклонение Python: {np.std(recon_py):.6f}")
    
    print(f"Максимум исходной: {np.max(original):.6f}")
    print(f"Максимум C++: {np.max(recon_cpp):.6f}")
    print(f"Максимум Python: {np.max(recon_py):.6f}")
    
    print(f"Минимум исходной: {np.min(original):.6f}")
    print(f"Минимум C++: {np.min(recon_cpp):.6f}")
    print(f"Минимум Python: {np.min(recon_py):.6f}")
    
    # Ошибки восстановления
    error_cpp = original - recon_cpp
    error_py = original - recon_py
    
    print(f"\n--- ОШИБКИ ВОССТАНОВЛЕНИЯ ---")
    print(f"Средняя абсолютная ошибка C++: {np.mean(np.abs(error_cpp)):.6f}")
    print(f"Средняя абсолютная ошибка Python: {np.mean(np.abs(error_py)):.6f}")
    
    print(f"Среднеквадратичная ошибка C++: {np.sqrt(np.mean(error_cpp**2)):.6f}")
    print(f"Среднеквадратичная ошибка Python: {np.sqrt(np.mean(error_py**2)):.6f}")
    
    # Относительные ошибки
    rel_error_cpp = np.abs(error_cpp) / (np.abs(original) + 1e-10)
    rel_error_py = np.abs(error_py) / (np.abs(original) + 1e-10)
    
    print(f"Средняя относительная ошибка C++: {np.mean(rel_error_cpp):.2e}")
    print(f"Средняя относительная ошибка Python: {np.mean(rel_error_py):.2e}")
    
    print(f"Максимальная относительная ошибка C++: {np.max(rel_error_cpp):.2e}")
    print(f"Максимальная относительная ошибка Python: {np.max(rel_error_py):.2e}")
    
    # Сравнение между C++ и Python восстановлениями
    diff_cpp_py = recon_cpp - recon_py
    print(f"\n--- СРАВНЕНИЕ C++ vs PYTHON ---")
    print(f"Средняя разность C++ - Python: {np.mean(diff_cpp_py):.2e}")
    print(f"Стандартное отклонение разности: {np.std(diff_cpp_py):.2e}")
    print(f"Максимальная абсолютная разность: {np.max(np.abs(diff_cpp_py)):.2e}")
    
    rel_diff_cpp_py = np.abs(diff_cpp_py) / (np.abs(recon_cpp) + 1e-10)
    print(f"Средняя относительная разность C++ vs Python: {np.mean(rel_diff_cpp_py):.2e}")
    print(f"Максимальная относительная разность C++ vs Python: {np.max(rel_diff_cpp_py):.2e}")
    
    return {
        'error_cpp': error_cpp,
        'error_py': error_py,
        'rel_error_cpp': rel_error_cpp,
        'rel_error_py': rel_error_py,
        'diff_cpp_py': diff_cpp_py,
        'rel_diff_cpp_py': rel_diff_cpp_py
    }

def plot_statistics_histograms(stft_stats, seis_stats, output_path):
    """Создает гистограммы статистических распределений."""
    print("Создание гистограмм статистических распределений...")
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Статистические распределения', fontsize=16)
    
    # Амплитудные спектры
    axs[0, 0].hist(stft_stats['amp_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[0, 0].hist(stft_stats['amp_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[0, 0].set_title('Распределение амплитуд STFT')
    axs[0, 0].set_xlabel('Амплитуда')
    axs[0, 0].set_ylabel('Плотность')
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')
    
    # Фазовые спектры
    axs[0, 1].hist(stft_stats['phase_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[0, 1].hist(stft_stats['phase_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[0, 1].set_title('Распределение фаз STFT')
    axs[0, 1].set_xlabel('Фаза (рад)')
    axs[0, 1].set_ylabel('Плотность')
    axs[0, 1].legend()
    
    # Относительные ошибки амплитуд
    axs[0, 2].hist(stft_stats['rel_error_amp'].flatten(), bins=50, alpha=0.7, density=True)
    axs[0, 2].set_title('Распределение относительных ошибок амплитуд')
    axs[0, 2].set_xlabel('Относительная ошибка')
    axs[0, 2].set_ylabel('Плотность')
    axs[0, 2].set_yscale('log')
    
    # Ошибки восстановления
    axs[1, 0].hist(seis_stats['error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[1, 0].hist(seis_stats['error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 0].set_title('Распределение ошибок восстановления')
    axs[1, 0].set_xlabel('Ошибка')
    axs[1, 0].set_ylabel('Плотность')
    axs[1, 0].legend()
    
    # Относительные ошибки восстановления
    axs[1, 1].hist(seis_stats['rel_error_cpp'].flatten(), bins=50, alpha=0.7, label='C++', density=True)
    axs[1, 1].hist(seis_stats['rel_error_py'].flatten(), bins=50, alpha=0.7, label='Python', density=True)
    axs[1, 1].set_title('Распределение относительных ошибок восстановления')
    axs[1, 1].set_xlabel('Относительная ошибка')
    axs[1, 1].set_ylabel('Плотность')
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()
    
    # Разность между C++ и Python
    axs[1, 2].hist(seis_stats['diff_cpp_py'].flatten(), bins=50, alpha=0.7, density=True)
    axs[1, 2].set_title('Распределение разности C++ - Python')
    axs[1, 2].set_xlabel('Разность')
    axs[1, 2].set_ylabel('Плотность')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"  Гистограммы сохранены в {output_path}")
    plt.close(fig)


# === 4. Основной блок выполнения ===
if __name__ == "__main__":
    print("Загрузка данных...")
    stft_cpp = load_stft_bin_cpp(CPP_STFT_FILE)
    stft_py = load_stft_bin_py(PY_STFT_FILE)
    original_seis = load_seismogram_bin(ORIGINAL_FILE, N_TRACES, N_SAMPLES)
    recon_cpp = load_seismogram_bin(CPP_RECON_FILE, N_TRACES, N_SAMPLES)
    recon_py = load_seismogram_bin(PY_RECON_FILE, N_TRACES, N_SAMPLES)

    print(f"Форма STFT C++: {stft_cpp.shape}")
    print(f"Форма STFT Python: {stft_py.shape}")
    if stft_cpp.shape != stft_py.shape:
        print("\n[ОШИБКА] Формы массивов STFT не совпадают! Проверьте функции загрузки.")
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

    stft_stats = analyze_stft_statistics(stft_cpp, stft_py, f_axis, t_axis)
    seis_stats = analyze_seismogram_statistics(original_seis, recon_cpp, recon_py)
    plot_statistics_histograms(stft_stats, seis_stats, os.path.join(OUTPUT_DIR, 'statistics_histograms.png'))

    print("\nАнализ завершен.")