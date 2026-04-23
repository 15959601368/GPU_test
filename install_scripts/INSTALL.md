# SAIGEQTL Installation Guide

## Quick Start (Source Installation)

**No conda/Miniforge required. No sudo required. Works on HPC clusters.**

```bash
git clone https://github.com/weizhou0/SAIGEQTL.git
cd SAIGEQTL/install_scripts
chmod +x *.sh
./install_from_source.sh
```

This will:
1. Build required C++ libraries (savvy, superlu, zstd, boost) from source
2. Install R package dependencies from CRAN
3. Install SAIGEQTL

---

## Prerequisites

### Required Tools

| Tool | Purpose | HPC Module Example |
|------|---------|-------------------|
| R ≥ 3.5.0 | Statistical computing | `module load R` |
| C++ compiler (GCC/Clang) | Compilation | `module load gcc` |
| CMake ≥ 3.10 | Build system | `module load cmake` |
| git | Download sources | Usually available |
| curl | Download sources | Usually available |

### Additional Requirements for macOS

macOS requires additional tools that are not needed on Linux:

```bash
# 1. Install Xcode command line tools (if not already installed)
xcode-select --install

# 2. Install gfortran (required for R packages)
#    Option A: From R project (recommended)
curl -LO https://mac.r-project.org/tools/gfortran-12.2-universal.pkg
sudo installer -pkg gfortran-12.2-universal.pkg -target /

#    Option B: Via Homebrew
brew install gcc

# 3. Install OpenMP support (for parallel processing)
brew install libomp
```

**Note:** Linux HPC clusters typically have these tools available via the module system (`module load gcc`).

---

## Installation Methods

### Method 1: One-Command Source Install (Recommended)

```bash
git clone https://github.com/weizhou0/SAIGEQTL.git
cd SAIGEQTL/install_scripts
./install_from_source.sh
```

### Method 2: Step-by-Step Source Install

If you prefer more control:

```bash
# Step 1: Clone the repository
git clone https://github.com/weizhou0/SAIGEQTL.git
cd SAIGEQTL/install_scripts

# Step 2: Build C++ dependencies (savvy, superlu, zstd, boost)
./install_dependencies.sh

# Step 3: Install R packages and SAIGEQTL
./install_R_packages.sh
```

---

## What Gets Built

The `install_dependencies.sh` script builds these libraries into `thirdParty/cget/`:

| Library | Version | Purpose |
|---------|---------|---------|
| zstd | 1.5.5 | Compression |
| SuperLU | 5.3.0 | Sparse linear solver |
| Boost | 1.82.0 | C++ utilities (headers only) |
| htslib | 1.19 | SAM/BAM/VCF library |
| savvy | latest | VCF/BCF file reading |

---

## Verifying Installation

```r
library(SAIGEQTL, lib.loc="/path/to/custom/library")
packageVersion("SAIGEQTL")
# Should print: [1] '0.3.2'
```

### Common Troubleshooting

### OpenMP errors on macOS
```bash
brew install libomp
```

### "bzip2 development files not found" (Linux)
This is OK - the script will automatically disable bz2 support if not available. Core functionality will still work. If you need full CRAM support, ask your system admin to install `bzip2-devel` or load the appropriate module.

### "lzma development files not found" (Linux)
Same as above - the script will disable lzma if not available. Load the xz module if available:
```bash
module load xz  # or liblzma
```

### Compilation errors with missing headers
Make sure `install_dependencies.sh` completed successfully. Check that `thirdParty/cget/include/` contains the required headers.

### "gfortran not found" (macOS)
R on macOS requires gfortran. Install from:
```bash
curl -LO https://mac.r-project.org/tools/gfortran-12.2-universal.pkg
sudo installer -pkg gfortran-12.2-universal.pkg -target /
```

## Getting Help

- Documentation: https://weizhou0.github.io/SAIGE-QTL-doc/
- Issues: https://github.com/weizhou0/SAIGEQTL/issues
