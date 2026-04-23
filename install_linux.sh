#!/bin/bash
# install_linux.sh
# Install SAIGEQTL with GPU support on Linux
#
# Prerequisites:
#   - R >= 3.5.0 with Rcpp, RcppArmadillo, RcppEigen, RcppParallel, RcppNumerical
#   - CUDA Toolkit >= 11.0 (nvcc, cuBLAS)
#   - For RTX 3080 Ti (Ampere): CUDA >= 11.1 is recommended
#
# Usage:
#   chmod +x install_linux.sh
#   ./install_linux.sh
#
# Or manually:
#   USE_GPU=1 R CMD INSTALL SAIGEQTL

set -e

echo "============================================"
echo "  SAIGEQTL GPU-Accelerated Build (Linux)"
echo "============================================"
echo ""

# ---- 1. Check CUDA ----
if [ -z "$CUDA_PATH" ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    else
        CUDA_PATH=/usr/local/cuda
    fi
fi

echo "[1/5] Checking CUDA..."
if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_PATH/bin/nvcc"
    echo "Please install CUDA Toolkit or set CUDA_PATH environment variable."
    echo ""
    echo "Download CUDA: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

NVCC_VERSION=$($CUDA_PATH/bin/nvcc --version | grep release | awk '{print $NF}')
echo "  CUDA path: $CUDA_PATH"
echo "  nvcc version: $NVCC_VERSION"

# ---- 2. Check R ----
echo ""
echo "[2/5] Checking R..."
if ! command -v R &> /dev/null; then
    echo "ERROR: R is not installed or not in PATH."
    exit 1
fi

R_VERSION=$(R --version | head -1)
echo "  $R_VERSION"

# ---- 3. Check R packages ----
echo ""
echo "[3/5] Checking required R packages..."

REQUIRED_PKGS="Rcpp RcppArmadillo RcppEigen RcppParallel RcppNumerical data.table dbplyr dplyr MASS Matrix methods nlme optparse RhpcBLASctl RSQLite"
MISSING_PKGS=""

for pkg in $REQUIRED_PKGS; do
    if ! Rscript -e "library($pkg, quietly=TRUE)" 2>/dev/null; then
        MISSING_PKGS="$MISSING_PKGS $pkg"
    fi
done

if [ -n "$MISSING_PKGS" ]; then
    echo "  WARNING: Missing R packages:$MISSING_PKGS"
    echo "  Installing missing packages..."
    Rscript -e "install.packages(c($(echo $MISSING_PKGS | tr ' ' ',' | sed 's/,$//')), repos='https://cloud.r-project.org')"
else
    echo "  All required R packages found."
fi

# ---- 4. Check GPU ----
echo ""
echo "[4/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU: $line"
    done
else
    echo "  WARNING: nvidia-smi not found. GPU may not be accessible."
fi

# ---- 5. Build & Install ----
echo ""
echo "[5/5] Building SAIGEQTL with GPU support..."
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set CUDA architecture for RTX 3080 Ti (sm_86)
# Also add sm_80 for broader compatibility
export CUDA_ARCH="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"

echo "  USE_GPU=1 CUDA_PATH=$CUDA_PATH R CMD INSTALL SAIGEQTL-main/"
USE_GPU=1 CUDA_PATH="$CUDA_PATH" R CMD INSTALL SAIGEQTL-main/

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "To verify GPU is working, run the test script:"
echo "  cd SAIGEQTL-main/extdata"
echo "  Rscript step1_fitNULLGLMM_qtl.R --useGPU TRUE ..."
echo ""
