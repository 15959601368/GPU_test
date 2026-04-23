/*
 * gpu_crossProd.h
 * Public C interface for GPU-accelerated GRM matrix-vector products.
 *
 * All functions return false (or are no-ops) when compiled without -DUSE_GPU,
 * so the rest of the codebase can call them unconditionally and still build
 * on machines without CUDA.
 */

#ifndef GPU_CROSSPROD_H
#define GPU_CROSSPROD_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * gpu_crossProd_init
 *   Upload the standardised genotype matrix G (N x M, column-major) to GPU.
 *   Must be called once before any compute calls.
 *   Returns true on success.
 */
bool gpu_crossProd_init(const float* G_host, int N, int M);

/*
 * gpu_crossProd_compute
 *   Compute K*b = (1/M) * G * G^T * b, storing the N-length result in out_host.
 *   scale is typically 1.0f/M.
 *   Returns true on success.
 */
bool gpu_crossProd_compute(const float* b_host,
                           float*       out_host,
                           int          N,
                           int          M,
                           float        scale);

/*
 * gpu_crossProd_compute_loco
 *   LOCO variant: (full_sum - chr_sum) / M_loco
 *   G on device is the FULL marker set; columns [start_idx..end_idx] belong
 *   to the chromosome being left out.
 *   scale is the desired final scale (typically 1.0).
 *   The function handles the internal LOCO scaling.
 *   Returns true on success.
 */
bool gpu_crossProd_compute_loco(const float* b_host,
                                float*       out_host,
                                int          N,
                                int          M_full,
                                int          start_idx,
                                int          end_idx,
                                float        scale);

/*
 * gpu_crossProd_cleanup
 *   Free all GPU resources. Safe to call at any time.
 */
void gpu_crossProd_cleanup(void);

/*
 * gpu_is_available
 *   Quick check: does the system have at least one CUDA-capable device?
 */
bool gpu_is_available(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPU_CROSSPROD_H */
