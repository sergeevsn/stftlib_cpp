#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <cassert>
#include "stft.hpp"

using namespace std;

#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>

template<typename T>
void read_binary_file(const std::string& filename, std::vector<std::vector<T>>& data, int n_rows, int n_cols) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open input file: " + filename);

    data.resize(n_rows, std::vector<T>(n_cols));
    for (int i = 0; i < n_rows; ++i) {
        in.read(reinterpret_cast<char*>(data[i].data()), n_cols * sizeof(T));
        if (!in) throw std::runtime_error("Error reading data at row " + std::to_string(i));
    }
}

template<typename T>
void write_binary_file(const std::string& filename, const std::vector<std::vector<T>>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output file: " + filename);

    for (const auto& row : data) {
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));
        if (!out) throw std::runtime_error("Error writing row");
    }
}


int main() {
    const string input_file = "data/input.bin";
    const string output_file = "data/output_cpp.bin";

    const int n_traces = 220;      // number of traces
    const int n_samples = 501;   // number of samples per trace
    const int frame_size = 64;
    const int hop_size = 32;
    const double dt = 0.004;      // time step (in seconds)

    vector<vector<float>> seismogram;
    read_binary_file<float>(input_file, seismogram, n_traces, n_samples);

    cout << "Read seismogram: " << n_traces << " traces, " << n_samples << " samples each\n";

    // Transform each trace using STFT
    vector<vector<vector<ComplexF>>> stft_data(n_traces);
    for (int i = 0; i < n_traces; ++i)
        stft_data[i] = STFT_forward_float(seismogram[i], frame_size, hop_size);

    // Output time and frequency lists
    int n_frames = stft_data[0].size();
    int n_freqs = frame_size / 2 + 1;

    cout << "\nTime axis (s): ";
    for (int t = 0; t < n_frames; ++t)
        cout << t * hop_size * dt << " ";
    cout << "\n";

    cout << "Frequency axis (Hz): ";
    for (int f = 0; f < n_freqs; ++f)
        cout << f / (dt * frame_size) << " ";
    cout << "\n";

    // Inverse STFT
    vector<vector<float>> reconstructed(n_traces);
    for (int i = 0; i < n_traces; ++i)
        reconstructed[i] = STFT_inverse_float(stft_data[i], frame_size, hop_size, n_samples);

    write_binary_file<float>(output_file, reconstructed);
    cout << "Reconstructed seismogram written to " << output_file << "\n";

    return 0;
}

