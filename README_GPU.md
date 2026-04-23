# SAIGEQTL GPU Acceleration

## Overview

This is a GPU-accelerated version of SAIGEQTL that speeds up **Step 1** (null GLMM fitting) by offloading the most expensive operation — GRM matrix-vector products (`parallelCrossProd`) — to NVIDIA GPUs using cuBLAS.

### Performance Target

The bottleneck in Step 1 is the PCG (Preconditioned Conjugate Gradient) solver, which calls `parallelCrossProd` hundreds of times. Each call computes:

```
K*b = (1/M) * G * G^T * b
```

where G is the N x M standardized genotype matrix.

| Operation | CPU (multi-threaded) | GPU (cuBLAS) | Expected Speedup |
|-----------|---------------------|--------------|-----------------|
| One PCG iteration | O(N * M) per marker | Two SGEMV calls | **5-20x** |
| Full Step 1 (N=5000, M=100k) | Hours | Minutes | **10-50x** |

### GPU: RTX 3080 Ti

- CUDA Cores: 10,240
- VRAM: 12 GB GDDR6X
- Compute Capability: 8.6 (Ampere)
- Required CUDA Version: >= 11.1

## Architecture

```
R: step1_fitNULLGLMM_qtl.R --useGPU TRUE
  → R: fitNULLGLMM_multiV(useGPU=TRUE)
    → setGPUforStep1(TRUE)  // uploads G to GPU once
      → C++: gpu_manager_init()
        → CUDA: gpu_crossProd_init()  // G stays on GPU
    → PCG iterations:
      → parallelCrossProd(bVec)
        → gpu_parallelCrossProd(bVec)  // cuBLAS SGEMV (fast!)
          → Step A: tmp = G^T * b    (one cuBLAS call)
          → Step B: out = G * tmp    (one cuBLAS call)
      → parallelCrossProd_LOCO(bVec)
        → gpu_parallelCrossProd_LOCO(bVec, start, end)
```

## Installation (Linux)

### Prerequisites

1. **CUDA Toolkit** >= 11.1
   - Download: https://developer.nvidia.com/cuda-downloads
   - Make sure `nvcc` is in PATH

2. **R** >= 3.5.0 with required packages:
   ```r
   install.packages(c("Rcpp", "RcppArmadillo", "RcppEigen", 
                       "RcppParallel", "RcppNumerical", "data.table",
                       "dbplyr", "dplyr", "MASS", "Matrix", "methods",
                       "nlme", "optparse", "RhpcBLASctl", "RSQLite"))
   ```

### Quick Install (Recommended)

```bash
cd SAIGEQTL
chmod +x install_linux.sh
./install_linux.sh
```

### Manual Install

```bash
# With GPU support
USE_GPU=1 R CMD INSTALL SAIGEQTL-main/

# Without GPU (identical to upstream)
R CMD INSTALL SAIGEQTL-main/

# Custom CUDA path
CUDA_PATH=/usr/local/cuda-12.1 USE_GPU=1 R CMD INSTALL SAIGEQTL-main/
```

### Custom CUDA Architecture

The default Makevars targets sm_80 and sm_86 (Ampere). For other GPUs:

- RTX 4090 / 4080 (Ada Lovelace): add `-gencode arch=compute_89,code=sm_89`
- RTX 3090 / 3080 (Ampere): default sm_86 is correct
- V100 / T4 (Volta): add `-gencode arch=compute_70,code=sm_70`
- A100 (Ampere): add `-gencode arch=compute_80,code=sm_80`

Edit `src/Makevars` to change the `NVCC_ARCH` variable.

## Usage

### Command Line (step1_fitNULLGLMM_qtl.R)

```bash
cd SAIGEQTL-main/extdata

Rscript step1_fitNULLGLMM_qtl.R \
  --plinkFile ../extdata/input/n.indep_100_n.cell_1 \
  --phenoFile ../extdata/input/pheno_1000samples.txt \
  --phenoCol pheno \
  --sampleIDColinphenoFile IID \
  --traitType quantitative \
  --outputPrefix ./test_output \
  --useGPU TRUE \
  --nThreads 1
```

### R API

```r
library(SAIGEQTL)

fitNULLGLMM_multiV(
  plinkFile = "path/to/plink",
  phenoFile = "path/to/pheno.txt",
  phenoCol = "pheno",
  sampleIDColinphenoFile = "IID",
  traitType = "quantitative",
  outputPrefix = "./output",
  useGPU = TRUE,      # <-- Enable GPU acceleration
  nThreads = 1
)
```

### Fallback Behavior

- If GPU is not available or initialization fails, the code **automatically falls back to CPU** without any errors.
- The CPU path is identical to the upstream SAIGEQTL implementation.

## Memory Requirements

The GPU stores the standardized genotype matrix G (N x M, float32):

| N (samples) | M (markers) | GPU Memory Needed |
|-------------|-------------|-------------------|
| 1,000 | 100,000 | ~0.4 GB |
| 5,000 | 100,000 | ~2.0 GB |
| 10,000 | 100,000 | ~4.0 GB |
| 10,000 | 500,000 | ~20.0 GB |

For the RTX 3080 Ti (12 GB VRAM), you can fit approximately:
- 10,000 samples x 250,000 markers, or
- 5,000 samples x 500,000 markers

If the matrix doesn't fit in VRAM, the system automatically falls back to CPU.

## Files Modified from Upstream

### New Files
- `src/gpu_crossProd.cu` — CUDA kernel + cuBLAS SGEMV implementation
- `src/gpu_crossProd.h` — C interface for GPU cross-product
- `src/gpu_manager.cpp` — Builds G matrix from NullGenoClass, manages GPU lifecycle
- `src/gpu_manager.h` — Public C++ interface
- `install_linux.sh` — Automated installation script
- `README_GPU.md` — This file

### Modified Files
- `src/Main.cpp` — Added GPU path in `parallelCrossProd`/`parallelCrossProd_LOCO`, added `setGPUforStep1`/`cleanupGPU`
- `src/Main.hpp` — Added declarations for new functions
- `src/RcppExports.cpp` — Registered new Rcpp functions
- `src/Makevars` — Conditional CUDA compilation (USE_GPU=1)
- `R/RcppExports.R` — Added R bindings for GPU functions
- `R/SAIGE_fitGLMM_fast_multiV.R` — Added `useGPU` parameter
- `extdata/step1_fitNULLGLMM_qtl.R` — Added `--useGPU` CLI flag
- `DESCRIPTION` — Updated SystemRequirements

## Troubleshooting

### "No CUDA device found"
- Check `nvidia-smi` — is the GPU visible?
- Check `nvcc --version` — is CUDA installed?
- Verify NVIDIA driver is up to date

### "Insufficient GPU memory"
- Reduce `--minMAFforGRM` to use fewer markers (smaller G matrix)
- Use a machine with more VRAM
- System will automatically fall back to CPU

### "cublasCreate failed"
- Check CUDA installation: `nvcc --version`
- Ensure cuBLAS is installed (part of CUDA toolkit)

### Compilation errors with nvcc
- Ensure R was built with a compatible compiler (same GCC version as CUDA)
- Try: `CUDA_PATH=/path/to/cuda USE_GPU=1 R CMD INSTALL SAIGEQTL-main/`

### Test with sample data
```bash
cd extdata
Rscript step1_fitNULLGLMM_qtl.R \
  --plinkFile input/n.indep_100_n.cell_1 \
  --phenoFile input/pheno_1000samples.txt \
  --phenoCol pheno \
  --sampleIDColinphenoFile IID \
  --traitType quantitative \
  --outputPrefix /tmp/saigeqtl_test \
  --useGPU TRUE \
  --LOCO FALSE \
  --nThreads 1
```
