#!/bin/bash
# SAIGEQTL - Build Third-Party Dependencies from Source
# No conda/Miniforge required. No sudo required.
# Works on HPC clusters with standard compilers.

set -e

echo "=============================================="
echo "SAIGEQTL: Building Dependencies from Source"
echo "=============================================="
echo ""
echo "This script builds required C++ libraries locally."
echo "No sudo or conda required."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Installation directory (in the package's thirdParty folder)
INSTALL_DIR="${PACKAGE_DIR}/thirdParty/cget"
mkdir -p "$INSTALL_DIR"/{include,lib,lib64}

echo "Package directory: $PACKAGE_DIR"
echo "Installing dependencies to: $INSTALL_DIR"
echo ""

# Temporary build directory
BUILD_DIR=$(mktemp -d)
echo "Build directory: $BUILD_DIR"
echo ""

# Detect OS
OS="$(uname -s)"
NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
echo "Detected OS: $OS"
echo "Using $NPROC parallel jobs"
echo ""

# Check for required tools
echo "[0/5] Checking for required tools..."
echo "----------------------------------------------"

for tool in cmake make git curl; do
    if ! command -v $tool &> /dev/null; then
        echo "ERROR: $tool is required but not found."
        echo "Please load the appropriate module or install it."
        echo "  e.g., module load cmake"
        exit 1
    fi
done

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    CXX=g++
    CC=gcc
elif command -v clang++ &> /dev/null; then
    CXX=clang++
    CC=clang
else
    echo "ERROR: No C++ compiler found (g++ or clang++)"
    exit 1
fi
echo "Using compiler: $CXX"

# Check compiler version
if command -v $CXX &> /dev/null; then
    CXX_VERSION=$($CXX --version | head -n 1)
    echo "Compiler version: $CXX_VERSION"

    # Check GCC version if using GCC
    if [[ "$CXX" == "g++" ]]; then
        GCC_MAJOR=$(echo | $CXX -dM -E - | grep __GNUC__ | awk '{print $3}')
        if [ -n "$GCC_MAJOR" ] && [ "$GCC_MAJOR" -lt 8 ]; then
            echo ""
            echo "WARNING: GCC version $GCC_MAJOR detected. GCC >= 8 recommended for C++14 support."
            echo "Consider loading a newer GCC module: module load gcc/10"
            echo ""
        fi
    fi
fi

# macOS-specific checks
if [ "$OS" = "Darwin" ]; then
    echo ""
    echo "=== macOS Requirements ==="

    # Check for gfortran (required for R packages)
    if [ ! -d "/opt/gfortran" ] && ! command -v gfortran &> /dev/null; then
        echo ""
        echo "WARNING: gfortran not found!"
        echo ""
        echo "R on macOS requires gfortran for compiling packages."
        echo "Please install it from the R project website:"
        echo ""
        echo "  # Download and install (requires sudo):"
        echo "  curl -LO https://mac.r-project.org/tools/gfortran-12.2-universal.pkg"
        echo "  sudo installer -pkg gfortran-12.2-universal.pkg -target /"
        echo ""
        echo "Or install via Homebrew:"
        echo "  brew install gcc"
        echo ""
        read -p "Continue anyway? [y/N] " response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "gfortran: OK"
    fi

    # Check for libomp
    OMP_FOUND=false
    for prefix in "/opt/homebrew" "/usr/local" "$HOME/homebrew"; do
        if [ -d "$prefix/opt/libomp" ]; then
            echo "libomp: OK ($prefix/opt/libomp)"
            OMP_FOUND=true
            break
        fi
    done
    if [ "$OMP_FOUND" = false ]; then
        echo ""
        echo "WARNING: libomp not found!"
        echo "Install with: brew install libomp"
        echo ""
    fi
fi
echo ""

# Function to check if library exists
check_lib() {
    local name=$1
    local header=$2
    if [ -f "$INSTALL_DIR/include/$header" ]; then
        echo "$name already installed, skipping..."
        return 0
    fi
    return 1
}

# 1. Build zstd
echo "[1/6] Building zstd (compression library)..."
echo "----------------------------------------------"
if check_lib "zstd" "zstd.h"; then
    : # skip
else
    cd "$BUILD_DIR"
    curl -L -o zstd.tar.gz https://github.com/facebook/zstd/releases/download/v1.5.5/zstd-1.5.5.tar.gz
    tar xzf zstd.tar.gz
    cd zstd-1.5.5
    make -j$NPROC PREFIX="$INSTALL_DIR" install
    echo "zstd installed successfully"
fi
echo ""

# 2. Build SuperLU
echo "[2/6] Building SuperLU (sparse linear solver)..."
echo "----------------------------------------------"
if check_lib "SuperLU" "superlu/slu_ddefs.h" || check_lib "SuperLU" "slu_ddefs.h"; then
    : # skip
else
    cd "$BUILD_DIR"
    # Use SuperLU 6.0.1 which has updated CMake support
    curl -L -o superlu.tar.gz https://github.com/xiaoyeli/superlu/archive/refs/tags/v6.0.1.tar.gz
    tar xzf superlu.tar.gz
    cd superlu-6.0.1
    mkdir -p build && cd build

    # Use system BLAS - different flags for macOS vs Linux
    if [ "$OS" = "Darwin" ]; then
        # macOS: use Accelerate framework
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
            -DCMAKE_C_COMPILER=$CC \
            -DCMAKE_BUILD_TYPE=Release \
            -Denable_internal_blaslib=OFF \
            -DTPL_BLAS_LIBRARIES="-framework Accelerate" \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    else
        # Linux: use system BLAS/LAPACK or OpenBLAS
        cmake .. \
            -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
            -DCMAKE_C_COMPILER=$CC \
            -DCMAKE_BUILD_TYPE=Release \
            -Denable_internal_blaslib=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    fi

    make -j$NPROC
    make install
    # Also copy headers to expected location
    mkdir -p "$INSTALL_DIR/include/superlu"
    cp "$INSTALL_DIR/include/slu_*.h" "$INSTALL_DIR/include/superlu/" 2>/dev/null || true
    echo "SuperLU installed successfully"
fi
echo ""

# 3. Build Boost (header-only parts we need)
echo "[3/6] Installing Boost headers..."
echo "----------------------------------------------"
if check_lib "Boost" "boost/math/distributions.hpp"; then
    : # skip
else
    cd "$BUILD_DIR"
    # Use Boost 1.83.0 from SourceForge (reliable mirror)
    echo "Downloading Boost (this may take a minute)..."
    curl -L -o boost.tar.bz2 "https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.bz2/download"
    tar xjf boost.tar.bz2
    cd boost_1_83_0
    # Just copy headers (we only need header-only libraries)
    cp -r boost "$INSTALL_DIR/include/"
    echo "Boost headers installed successfully"
fi
echo ""

# 4. Build htslib (required by savvy)
echo "[4/6] Building htslib (for savvy)..."
echo "----------------------------------------------"
if check_lib "htslib" "htslib/hts.h"; then
    : # skip
else
    cd "$BUILD_DIR"
    curl -L -o htslib.tar.bz2 https://github.com/samtools/htslib/releases/download/1.19/htslib-1.19.tar.bz2
    tar xjf htslib.tar.bz2
    cd htslib-1.19

    # Check for optional compression libraries
    HTSLIB_OPTS="--prefix=$INSTALL_DIR --disable-libcurl --disable-gcs --disable-s3"

    # Check for bzip2
    if ! pkg-config --exists bzip2 2>/dev/null && \
       ! (echo '#include <bzlib.h>' | $CC -E - >/dev/null 2>&1); then
        echo "Note: bzip2 development files not found, disabling bz2 support"
        HTSLIB_OPTS="$HTSLIB_OPTS --disable-bz2"
    fi

    # Check for lzma
    if ! pkg-config --exists liblzma 2>/dev/null && \
       ! (echo '#include <lzma.h>' | $CC -E - >/dev/null 2>&1); then
        echo "Note: lzma development files not found, disabling lzma support"
        HTSLIB_OPTS="$HTSLIB_OPTS --disable-lzma"
    fi

    echo "Configuring htslib with: $HTSLIB_OPTS"
    ./configure $HTSLIB_OPTS
    make -j$NPROC
    make install
    echo "htslib installed successfully"
fi
echo ""

# 5. Build shrinkwrap (required by savvy)
echo "[5/6] Building shrinkwrap (compression library)..."
echo "----------------------------------------------"
if check_lib "shrinkwrap" "shrinkwrap/zstd.hpp"; then
    : # skip
else
    cd "$BUILD_DIR"
    git clone --depth 1 https://github.com/jonathonl/shrinkwrap.git
    cd shrinkwrap
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_CXX_COMPILER=$CXX \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DZSTD_LIBRARY="$INSTALL_DIR/lib/libzstd.a" \
        -DZSTD_INCLUDE_DIR="$INSTALL_DIR/include"
    make -j$NPROC
    make install
    echo "shrinkwrap installed successfully"
fi
echo ""

# 6. Build savvy
echo "[6/6] Building savvy (VCF/BCF library)..."
echo "----------------------------------------------"
if check_lib "savvy" "savvy/reader.hpp"; then
    : # skip
else
    cd "$BUILD_DIR"
    git clone --depth 1 https://github.com/statgen/savvy.git
    cd savvy
    mkdir -p build && cd build

    # Find htslib - could be .a or .so/.dylib
    if [ -f "$INSTALL_DIR/lib/libhts.a" ]; then
        HTS_LIB="$INSTALL_DIR/lib/libhts.a"
    elif [ -f "$INSTALL_DIR/lib/libhts.so" ]; then
        HTS_LIB="$INSTALL_DIR/lib/libhts.so"
    elif [ -f "$INSTALL_DIR/lib/libhts.dylib" ]; then
        HTS_LIB="$INSTALL_DIR/lib/libhts.dylib"
    else
        echo "ERROR: htslib not found. Check step 4."
        exit 1
    fi

    # Set library paths for pkg-config and cmake
    export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$INSTALL_DIR/lib:$LIBRARY_PATH"
    export CPATH="$INSTALL_DIR/include:$CPATH"

    # We only need the headers from savvy, not the sav executable
    # So we'll just install the headers directly instead of building the full project
    echo "Installing savvy headers..."
    mkdir -p "$INSTALL_DIR/include/savvy"
    cp ../include/savvy/*.hpp "$INSTALL_DIR/include/savvy/"

    # Create a minimal cmake config for savvy
    mkdir -p "$INSTALL_DIR/share/savvy"
    cat > "$INSTALL_DIR/share/savvy/savvy-config.cmake" << 'SAVVYCMAKE'
# Minimal savvy config - header-only installation
set(SAVVY_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../include")
set(SAVVY_FOUND TRUE)
SAVVYCMAKE

    echo "savvy headers installed successfully"
fi
echo ""

# Cleanup
echo "Cleaning up build directory..."
rm -rf "$BUILD_DIR"

echo ""
echo "=============================================="
echo "Dependencies installed successfully!"
echo "=============================================="
echo ""
echo "Installed to: $INSTALL_DIR"
echo ""
echo "Next step: Run install_R_packages.sh to install SAIGEQTL"
echo ""
