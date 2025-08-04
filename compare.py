import numpy as np
import matplotlib.pyplot as plt

# === Parameters ===
n_traces = 220
n_samples = 501
input_file = "data/input.bin"
cpp_file = "data/output_cpp.bin"
py_file = "data/output_py.bin"

# === Load binary files ===
def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(n_traces, n_samples)

data_in = load_bin(input_file)
data_cpp = load_bin(cpp_file)
data_py = load_bin(py_file)

# === Common color scale ===
vmin, vmax = np.percentile(data_in, [2, 98])


# === Display ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

titles = ["Original", "C++ ISTFT", "Python ISTFT"]
datas = [data_in, data_cpp, data_py]

for ax, title, dat in zip(axs, titles, datas):
    im = ax.imshow(dat.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Trace")


fig.suptitle("STFT Reconstruction Comparison", fontsize=14)
plt.tight_layout()
plt.show()
