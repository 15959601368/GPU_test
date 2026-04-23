#!/bin/bash
# SAIGEQTL - Complete Source Installation
# No conda/Miniforge required. No sudo required.
# Works on HPC clusters.

set -e

echo "=============================================="
echo "SAIGEQTL: Complete Source Installation"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Build required C++ libraries from source"
echo "  2. Install R package dependencies"
echo "  3. Install SAIGEQTL"
echo ""
echo "No conda or sudo required."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Build dependencies
echo "========== Step 1: Building C++ Dependencies =========="
bash "$SCRIPT_DIR/install_dependencies.sh"

# Step 2: Install R packages
echo ""
echo "========== Step 2: Installing R Packages =========="
# Allow overriding R library dir before running full installer:
#   export SAIGEQTL_R_LIB=/some/writable/path
R_LIB_PROP="${SAIGEQTL_R_LIB:-${R_LIBS_USER:-$HOME/R/library}}"
echo "R library directory (can override with SAIGEQTL_R_LIB): $R_LIB_PROP"
export SAIGEQTL_R_LIB="$R_LIB_PROP"

bash "$SCRIPT_DIR/install_R_packages.sh"

