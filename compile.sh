#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
KFR_INCLUDE="$ROOT/kfr/include"
KFR_LIB="$ROOT/kfr/build/lib"

CXX="${CXX:-g++}"

build_kfr() {
    echo "Building KFR 6.2.0 (GCC, single-arch)..."
    if [[ ! -f "$ROOT/kfr/include/kfr/dft.hpp" ]]; then
        echo "KFR submodule not initialized. Run: git submodule update --init kfr"
        exit 1
    fi
    (
        cd "$ROOT/kfr"
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
            -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
            -DKFR_ENABLE_DFT=ON -DKFR_ENABLE_MULTIARCH=OFF
        cmake --build build -j"$(nproc)"
    )
}

if [[ ! -f "$ROOT/kfr/include/kfr/dft.hpp" ]]; then
    echo "Initializing KFR submodule..."
    git -C "$ROOT" submodule update --init kfr
fi

if [[ ! -f "$KFR_LIB/libkfr_dft.a" ]]; then
    build_kfr
fi

"$CXX" -std=c++17 -O3 -march=native \
    cpp_stft.cpp -I. -I"$KFR_INCLUDE" \
    -L"$KFR_LIB" -lkfr_dft \
    -lpthread \
    -o cpp_stft
