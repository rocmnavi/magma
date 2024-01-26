/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel
*/
#include "magma_trisolve.h"


void magma_trisolve_free(magma_solve_info_t *solve_info) {
#if CUDA_VERSION >= 11031
    if (solve_info->descr) {
        hipsparseSpSM_destroyDescr(solve_info->descr);
        if (solve_info->buffer)
            magma_free(solve_info->buffer);
        solve_info->buffer = NULL;
        solve_info->descr = NULL;
    }
#else
    if (solve_info->descr) {
        hipsparseDestroyCsrsm2Info(solve_info->descr);
        solve_info->descr = NULL;
    }
#endif
}
