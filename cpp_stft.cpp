#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <cmath>
#include <limits>
#include <stdexcept>

// Include our new header-only library
#include "stft.hpp"

// File handling functions
template<typename T>
void read_binary_file(const std::string& filename, std::vector<std::vector<T>>& data, int n_rows, int n_cols) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open input file: " + filename);

    data.assign(n_rows, std::vector<T>(n_cols));
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

template<typename T>
void write_stft_binary_file(const std::string& filename, const std::vector<std::vector<std::vector<std::complex<T>>>>& stft_data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open STFT output file: " + filename);

    if (stft_data.empty() || stft_data[0].empty() || stft_data[0][0].empty()) {
        std::cerr << "Warning: STFT data is empty, writing an empty file." << std::endl;
        return;
    }

    int n_traces = stft_data.size();
    int n_frames = stft_data[0].size();
    int n_freqs = stft_data[0][0].size();

    out.write(reinterpret_cast<const char*>(&n_traces), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_frames), sizeof(int));
    out.write(reinterpret_cast<const char*>(&n_freqs), sizeof(int));

    for (int trace = 0; trace < n_traces; ++trace) {
        for (int frame = 0; frame < n_frames; ++frame) {
            out.write(reinterpret_cast<const char*>(stft_data[trace][frame].data()), n_freqs * sizeof(std::complex<T>));
             if (!out) throw std::runtime_error("Error writing STFT data for trace " + std::to_string(trace));
        }
    }
}


int main() {
    // Parameters
    using RealType = float;

    const std::string input_file = "data/input.bin";
    const std::string output_file = "data/output_cpp.bin";
    const std::string stft_output_file = "data/stft_cpp.bin";

    const int n_traces = 220;
    const int n_samples = 501;
    const int frame_size = 64;
    const int hop_size = 32;
    const double dt = 0.004;

    const BoundaryType boundary_type = BoundaryType::EVEN;

    // Read data
    std::vector<std::vector<RealType>> seismogram;
    try {
        read_binary_file<RealType>(input_file, seismogram, n_traces, n_samples);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Read seismogram: " << seismogram.size() << " traces, " << seismogram[0].size() << " samples each\n";
    std::cout << "Using boundary type: " << (boundary_type == BoundaryType::ZERO ? "ZERO" : "EVEN") << "\n";

    // Forward STFT
    std::vector<std::vector<std::vector<std::complex<RealType>>>> stft_data(n_traces);
    for (int i = 0; i < n_traces; ++i) {
        stft_data[i] = STFT_forward<RealType>(seismogram[i], frame_size, hop_size, boundary_type);
    }

    write_stft_binary_file<RealType>(stft_output_file, stft_data);
    std::cout << "STFT data (raw) written to " << stft_output_file << "\n";

    // Inverse STFT
    std::vector<std::vector<RealType>> reconstructed(n_traces);
    for (int i = 0; i < n_traces; ++i) {
        reconstructed[i] = STFT_inverse<RealType>(stft_data[i], frame_size, hop_size, n_samples, boundary_type);
    }

    write_binary_file<RealType>(output_file, reconstructed);
    std::cout << "Reconstructed seismogram written to " << output_file << "\n";

    // Quality analysis
    std::cout << "\n=== ANALYSIS ===\n";
    double total_diff_sum = 0.0;
    double max_diff = 0.0;
    double total_squared_diff = 0.0;
    
    for (int trace = 0; trace < n_traces; ++trace) {
        for (int sample = 0; sample < n_samples; ++sample) {
            double diff = std::abs(static_cast<double>(seismogram[trace][sample]) - static_cast<double>(reconstructed[trace][sample]));
            total_diff_sum += diff;
            max_diff = std::max(max_diff, diff);
            total_squared_diff += diff * diff;
        }
    }
    
    double mean_diff = total_diff_sum / (n_traces * n_samples);
    double rms_diff = std::sqrt(total_squared_diff / (n_traces * n_samples));
    
    std::cout << "Reconstruction vs Original:\n";
    std::cout << "  Max absolute difference: " << max_diff << "\n";
    std::cout << "  Mean absolute difference: " << mean_diff << "\n";
    std::cout << "  RMS difference: " << rms_diff << "\n";

    return 0;
}