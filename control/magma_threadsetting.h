/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Azzam Haidar
*/

#ifndef MAGMA_THREADSETTING_H
#define MAGMA_THREADSETTING_H

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Internal routines

void magma_set_omp_numthreads(magma_int_t numthreads);
void magma_set_lapack_numthreads(magma_int_t numthreads);
magma_int_t magma_get_lapack_numthreads();
magma_int_t magma_get_parallel_numthreads();
magma_int_t magma_get_omp_numthreads();

#ifdef __cplusplus
}
#endif

#endif  // MAGMA_THREADSETTING_H
