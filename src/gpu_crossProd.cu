/*
 * gpu_crossProd.cu
 * GPU-accelerated GRM matrix-vector product for SAIGEQTL Step 1
 *
 * Strategy:
 *   parallelCrossProd computes: K*b = (1/M) * G * G^T * b
 *   where G is N x M standardized genotype matrix.
 *
 *   GPU path:
 *     1. Pre-upload G to GPU device memory (done once per model fit).
 *     2. Each PCG iteration: two cuBLAS SGEMV calls:
 *        step A: tmp = G^T * b   (M-vector)
 *        step B: out = G * tmp   (N-vector), then divide by M.
 *
 * Build requirement:
 *   nvcc + cuBLAS.  Compile flag: -DUSE_GPU
 *   If USE_GPU is not defined the file exposes only CPU stubs so the
 *   package still builds on machines without CUDA.
 */

#include "gpu_crossProd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* GPU implementation                                                   */
/* ------------------------------------------------------------------ */
#ifdef USE_GPU

#include <cuda_runtime.h>
#include <cublas_v2.h>

/* ---------- helper macros ----------------------------------------- */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA] Error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            gpu_crossProd_cleanup();                                      \
            return false;                                                 \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t _s = (call);                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "[cuBLAS] Error %s:%d: status=%d\n",         \
                    __FILE__, __LINE__, (int)_s);                         \
            gpu_crossProd_cleanup();                                      \
            return false;                                                 \
        }                                                                 \
    } while (0)

/* ---------- module-level state ------------------------------------ */
static cublasHandle_t g_handle        = nullptr;
static float*         g_d_G           = nullptr;  /* device G: N x M (col-major) */
static float*         g_d_b           = nullptr;  /* device b: N */
static float*         g_d_tmp         = nullptr;  /* device tmp: M */
static float*         g_d_out         = nullptr;  /* device out: N */
static int            g_N             = 0;
static int            g_M             = 0;
static bool           g_gpu_ready     = false;

/* ------------------------------------------------------------------ */
bool gpu_crossProd_init(const float* G_host, int N, int M)
{
    /* Already initialised with same dimensions – skip re-upload */
    if (g_gpu_ready && g_N == N && g_M == M) {
        /* Just re-upload G in case markers changed (LOCO) */
        cudaError_t ce = cudaMemcpy(g_d_G, G_host,
                                    (size_t)N * M * sizeof(float),
                                    cudaMemcpyHostToDevice);
        return (ce == cudaSuccess);
    }

    /* Clean up previous state */
    gpu_crossProd_cleanup();

    /* Select device */
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        fprintf(stderr, "[GPU] No CUDA device found, falling back to CPU.\n");
        return false;
    }
    cudaSetDevice(0);

    /* Print device info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "[GPU] Using device: %s  (%.1f GB VRAM)\n",
            prop.name, prop.totalGlobalMem / 1.0e9);

    size_t need = (size_t)N * M * sizeof(float)   /* G   */
                + (size_t)N   * sizeof(float)       /* b   */
                + (size_t)M   * sizeof(float)       /* tmp */
                + (size_t)N   * sizeof(float);      /* out */

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "[GPU] Memory needed: %.2f GB,  free: %.2f GB\n",
            need / 1.0e9, free_mem / 1.0e9);

    if (need > free_mem * 0.90) {
        fprintf(stderr, "[GPU] Insufficient GPU memory – falling back to CPU.\n");
        return false;
    }

    /* Create cuBLAS handle */
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[GPU] cublasCreate failed.\n");
        return false;
    }

    /* Allocate device arrays */
    CUDA_CHECK(cudaMalloc(&g_d_G,   (size_t)N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_b,   (size_t)N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_tmp, (size_t)M     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d_out, (size_t)N     * sizeof(float)));

    /* Upload G (host: row-major N x M, device: col-major N x M treated as Fortran) */
    /* G_host is stored marker-by-marker (each column is one SNP std geno vector)   */
    CUDA_CHECK(cudaMemcpy(g_d_G, G_host,
                          (size_t)N * M * sizeof(float),
                          cudaMemcpyHostToDevice));

    g_N = N;  g_M = M;  g_gpu_ready = true;
    fprintf(stderr, "[GPU] Genotype matrix uploaded: N=%d, M=%d  (%.2f GB)\n",
            N, M, (double)N * M * sizeof(float) / 1.0e9);
    return true;
}

/* ------------------------------------------------------------------ */
bool gpu_crossProd_compute(const float* b_host,
                           float*       out_host,
                           int          N,
                           int          M,
                           float        scale)
{
    if (!g_gpu_ready) return false;

    /* Upload b */
    CUDA_CHECK(cudaMemcpy(g_d_b, b_host,
                          (size_t)N * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudaMemset(g_d_tmp, 0, (size_t)M * sizeof(float));
    cudaMemset(g_d_out, 0, (size_t)N * sizeof(float));

    const float alpha = 1.0f, beta = 0.0f;

    /*
     * Step A:  tmp = G^T * b
     *   G is N×M stored column-major → cublasSgemv with CUBLAS_OP_T
     *   treats it as M×N, so "transpose" gives N×M view → multiply by b (N).
     *   Actually: G col-major N×M means leading dim = N.
     *   Op = T  ⇒  y = G^T * x  where x has dim N → y has dim M.  ✓
     */
    CUBLAS_CHECK(cublasSgemv(g_handle, CUBLAS_OP_T,
                             N, M,          /* rows, cols of G */
                             &alpha,
                             g_d_G, N,      /* lda = N */
                             g_d_b, 1,
                             &beta,
                             g_d_tmp, 1));

    /*
     * Step B:  out = G * tmp
     *   Op = N  ⇒  y = G * x  where x has dim M → y has dim N.  ✓
     */
    CUBLAS_CHECK(cublasSgemv(g_handle, CUBLAS_OP_N,
                             N, M,
                             &alpha,
                             g_d_G, N,
                             g_d_tmp, 1,
                             &beta,
                             g_d_out, 1));

    /* Download result */
    CUDA_CHECK(cudaMemcpy(out_host, g_d_out,
                          (size_t)N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Apply scale = 1/M */
    for (int i = 0; i < N; i++) out_host[i] *= scale;

    return true;
}

/* ------------------------------------------------------------------ */
bool gpu_crossProd_compute_loco(const float* b_host,
                                float*       out_host,
                                int          N,
                                int          M_full,
                                int          start_idx,
                                int          end_idx,
                                float        scale)
{
    if (!g_gpu_ready) return false;

    /* LOCO = full crossProd - crossProd of [startIdx, endIdx] chromosome */
    /* We compute full result first */
    bool ok = gpu_crossProd_compute(b_host, out_host, N, M_full, 1.0f);
    if (!ok) return false;

    /* Number of markers in the excluded chromosome */
    int M_chr = end_idx - start_idx + 1;
    if (M_chr <= 0) {
        /* Nothing to subtract */
        for (int i = 0; i < N; i++) out_host[i] *= scale;
        return true;
    }

    /* Temporary host buffer for chromosome contribution */
    std::vector<float> chr_out(N, 0.0f);

    /* Pointer to the chromosome columns in device G */
    const float* d_G_chr = g_d_G + (size_t)start_idx * N;

    /* tmp_chr = G_chr^T * b */
    static float* d_tmp_chr = nullptr;
    static int    d_tmp_chr_size = 0;
    if (d_tmp_chr_size < M_chr) {
        if (d_tmp_chr) cudaFree(d_tmp_chr);
        cudaMalloc(&d_tmp_chr, (size_t)M_chr * sizeof(float));
        d_tmp_chr_size = M_chr;
    }

    static float* d_out_chr = nullptr;
    static int    d_out_chr_size = 0;
    if (d_out_chr_size < N) {
        if (d_out_chr) cudaFree(d_out_chr);
        cudaMalloc(&d_out_chr, (size_t)N * sizeof(float));
        d_out_chr_size = N;
    }

    const float alpha = 1.0f, beta = 0.0f;

    /* Upload b */
    cudaMemcpy(g_d_b, b_host, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    /* Step A: tmp_chr = G_chr^T * b */
    cublasSgemv(g_handle, CUBLAS_OP_T,
                N, M_chr,
                &alpha,
                d_G_chr, N,
                g_d_b, 1,
                &beta,
                d_tmp_chr, 1);

    /* Step B: out_chr = G_chr * tmp_chr */
    cublasSgemv(g_handle, CUBLAS_OP_N,
                N, M_chr,
                &alpha,
                d_G_chr, N,
                d_tmp_chr, 1,
                &beta,
                d_out_chr, 1);

    cudaMemcpy(chr_out.data(), d_out_chr,
               (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);

    /* LOCO result: (full_sum - chr_sum) / M_loco */
    int M_loco = M_full - M_chr;
    float loco_scale = (M_loco > 0) ? (1.0f / M_loco) : scale;
    for (int i = 0; i < N; i++) {
        out_host[i] -= chr_out[i];
        out_host[i] *= loco_scale;
    }

    return true;
}

/* ------------------------------------------------------------------ */
void gpu_crossProd_cleanup()
{
    if (g_handle)  { cublasDestroy(g_handle);  g_handle  = nullptr; }
    if (g_d_G)     { cudaFree(g_d_G);          g_d_G     = nullptr; }
    if (g_d_b)     { cudaFree(g_d_b);          g_d_b     = nullptr; }
    if (g_d_tmp)   { cudaFree(g_d_tmp);        g_d_tmp   = nullptr; }
    if (g_d_out)   { cudaFree(g_d_out);        g_d_out   = nullptr; }
    g_N = 0;  g_M = 0;  g_gpu_ready = false;
}

/* ------------------------------------------------------------------ */
bool gpu_is_available()
{
    int ndev = 0;
    return (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0);
}

/* ------------------------------------------------------------------ */
/* CPU stubs when USE_GPU is defined but GPU init failed              */
/* (already handled by returning false – caller falls back to CPU)    */
/* ------------------------------------------------------------------ */

#else  /* !USE_GPU */

/* ------------------------------------------------------------------ */
/* Pure CPU stubs – compiled when CUDA toolkit not available           */
/* ------------------------------------------------------------------ */

#include <vector>

bool gpu_crossProd_init(const float* /*G_host*/, int /*N*/, int /*M*/)
{
    return false;
}

bool gpu_crossProd_compute(const float* /*b_host*/,
                           float*       /*out_host*/,
                           int          /*N*/,
                           int          /*M*/,
                           float        /*scale*/)
{
    return false;
}

bool gpu_crossProd_compute_loco(const float* /*b_host*/,
                                float*       /*out_host*/,
                                int          /*N*/,
                                int          /*M_full*/,
                                int          /*start_idx*/,
                                int          /*end_idx*/,
                                float        /*scale*/)
{
    return false;
}

void gpu_crossProd_cleanup() {}

bool gpu_is_available() { return false; }

#endif  /* USE_GPU */
