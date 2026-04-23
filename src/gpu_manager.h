#pragma once
/*
 * gpu_manager.h
 * Public C++ interface to the GPU manager.
 */

#define ARMA_USE_SUPERLU 1
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

/*
 * gpu_manager_init
 *   Must be called AFTER NullGenoClass has been initialised (i.e. after
 *   setgenoNULL / setGenoObj have populated ptr_gNULLGENOobj).
 *   user_requested: pass true if --useGPU=TRUE was specified.
 *   Returns true on success.
 */
bool gpu_manager_init(bool user_requested);

/*
 * gpu_manager_update_loco
 *   Call after updating the LOCO chromosome index so caches are refreshed.
 */
void gpu_manager_update_loco();

/*
 * gpu_manager_is_ready
 *   Returns true when the GPU is initialised and ready for compute.
 */
bool gpu_manager_is_ready();

/*
 * gpu_manager_cleanup
 *   Free GPU resources. Safe to call at any time.
 */
void gpu_manager_cleanup();

/*
 * GPU drop-in replacements.
 * Return an empty fvec when GPU is unavailable (caller must fall back).
 */
arma::fvec gpu_parallelCrossProd(const arma::fcolvec& bVec);
arma::fvec gpu_parallelCrossProd_LOCO(const arma::fcolvec& bVec,
                                       int start_idx, int end_idx);
