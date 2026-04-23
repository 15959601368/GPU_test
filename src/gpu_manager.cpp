/*
 * gpu_manager.cpp
 * GPU state manager for SAIGEQTL.
 *
 * Responsibilities:
 *   1. Build the dense standardised-genotype matrix G from NullGenoClass.
 *   2. Upload G to GPU once before Step-1 iteration starts.
 *   3. Provide gpu_parallelCrossProd / gpu_parallelCrossProd_LOCO as
 *      drop-in replacements for the TBB Workers in Main.cpp.
 *   4. Fall back transparently to the CPU path when GPU is unavailable.
 *
 * Compile with -DUSE_GPU to activate the GPU path.
 */

#define ARMA_USE_SUPERLU 1
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include "gpu_manager.h"
#include "gpu_crossProd.h"
#include "GENO_null.hpp"

#include <vector>
#include <iostream>
#include <cassert>

/* ------------------------------------------------------------------ */
/* Module state                                                         */
/* ------------------------------------------------------------------ */
static bool   g_gpu_manager_ready = false;
static int    g_N_gpu             = 0;    /* Nnomissing */
static int    g_M_gpu             = 0;    /* number of GRM markers */
static bool   g_gpu_enabled       = false;/* set at R level */

/* Reference to the global NullGenoClass object (defined in Main.cpp) */
extern NullGENO::NullGenoClass* ptr_gNULLGENOobj;

/* ------------------------------------------------------------------ */
/* Internal: build dense G matrix on host                              */
/* ------------------------------------------------------------------ */
static std::vector<float> build_G_matrix(int N, int M_grm)
{
    std::vector<float> G(static_cast<size_t>(N) * M_grm);

    arma::fvec vec;
    int col = 0;
    int total = ptr_gNULLGENOobj->getnumberofMarkerswithMAFge_minMAFtoConstructGRM();

    for (int m = 0; m < total; ++m) {
        int ok = ptr_gNULLGENOobj->Get_OneSNP_StdGeno(m, &vec);
        if (ok == 0) continue;                /* marker filtered */
        if (col >= M_grm) break;

        /* vec has length N – copy into column col of G */
        float* dst = G.data() + static_cast<size_t>(col) * N;
        for (int i = 0; i < N; ++i) dst[i] = vec[i];
        ++col;
    }

    if (col != M_grm) {
        Rcpp::warning("[GPU Manager] Expected %d GRM markers but found %d",
                      M_grm, col);
        G.resize(static_cast<size_t>(N) * col);
    }

    return G;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

/* Called from R via setGPUforStep1() after NullGenoClass is ready     */
bool gpu_manager_init(bool user_requested)
{
    g_gpu_enabled = user_requested;
    g_gpu_manager_ready = false;

    if (!user_requested) {
        Rprintf("[GPU Manager] GPU mode not requested – using CPU.\n");
        return false;
    }

    if (!gpu_is_available()) {
        Rprintf("[GPU Manager] No CUDA device detected – falling back to CPU.\n");
        return false;
    }

    if (ptr_gNULLGENOobj == nullptr) {
        Rprintf("[GPU Manager] NullGenoClass not initialised yet.\n");
        return false;
    }

    int N = ptr_gNULLGENOobj->getNnomissing();
    int M = ptr_gNULLGENOobj->getnumberofMarkerswithMAFge_minMAFtoConstructGRM();

    if (N <= 0 || M <= 0) {
        Rprintf("[GPU Manager] Invalid dimensions N=%d M=%d\n", N, M);
        return false;
    }

    Rprintf("[GPU Manager] Building genotype matrix: N=%d, M=%d  (%.2f GB) ...\n",
            N, M, (double)N * M * 4 / 1e9);

    std::vector<float> G = build_G_matrix(N, M);
    int M_actual = static_cast<int>(G.size()) / N;

    bool ok = gpu_crossProd_init(G.data(), N, M_actual);
    if (!ok) {
        Rprintf("[GPU Manager] GPU init failed – falling back to CPU.\n");
        return false;
    }

    g_N_gpu = N;
    g_M_gpu = M_actual;
    g_gpu_manager_ready = true;

    Rprintf("[GPU Manager] GPU ready! (N=%d, M=%d)\n", N, M_actual);
    return true;
}

/* After LOCO chromosome switch – need to re-upload unchanged G but    */
/* the index range changes (handled inside gpu_crossProd_compute_loco) */
void gpu_manager_update_loco()
{
    /* Nothing extra needed – LOCO is handled by passing start/end idx  */
}

bool gpu_manager_is_ready()
{
    return g_gpu_manager_ready;
}

void gpu_manager_cleanup()
{
    gpu_crossProd_cleanup();
    g_gpu_manager_ready = false;
}

/* ------------------------------------------------------------------ */
/* Drop-in replacements for parallelCrossProd / parallelCrossProd_LOCO */
/* Returns empty vector if GPU not ready (caller falls back to CPU)    */
/* ------------------------------------------------------------------ */

arma::fvec gpu_parallelCrossProd(const arma::fcolvec& bVec)
{
    if (!g_gpu_manager_ready) return arma::fvec();

    int N = g_N_gpu;
    int M = g_M_gpu;
    arma::fvec out(N, arma::fill::zeros);

    bool ok = gpu_crossProd_compute(bVec.memptr(),
                                    out.memptr(),
                                    N, M,
                                    1.0f / static_cast<float>(M));
    if (!ok) {
        Rprintf("[GPU Manager] Compute failed – will fall back to CPU.\n");
        g_gpu_manager_ready = false;
        return arma::fvec();
    }
    return out;
}

arma::fvec gpu_parallelCrossProd_LOCO(const arma::fcolvec& bVec,
                                       int start_idx, int end_idx)
{
    if (!g_gpu_manager_ready) return arma::fvec();

    int N = g_N_gpu;
    int M = g_M_gpu;
    arma::fvec out(N, arma::fill::zeros);

    bool ok = gpu_crossProd_compute_loco(bVec.memptr(),
                                         out.memptr(),
                                         N, M,
                                         start_idx, end_idx,
                                         1.0f);  /* scale applied inside */
    if (!ok) {
        Rprintf("[GPU Manager] LOCO compute failed – will fall back to CPU.\n");
        g_gpu_manager_ready = false;
        return arma::fvec();
    }
    return out;
}
