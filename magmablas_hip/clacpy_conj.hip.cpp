#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zlacpy_conj.hip.cpp, normal z -> c, Mon Oct 28 11:12:05 2024

*/
#include "magma_internal.h"

#define BLOCK_SIZE 64

/******************************************************************************/
// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old cswap routine. Update to new clacpy code for 2D matrix?

__global__ void clacpy_conj_kernel(
    int n,
    magmaFloatComplex *A1, int lda1,
    magmaFloatComplex *A2, int lda2 )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_C_CONJ( A1[offset1] );
    }
}


/******************************************************************************/
extern "C" void 
magmablas_clacpy_conj(
    magma_int_t n,
    magmaFloatComplex_ptr dA1, magma_int_t lda1, 
    magmaFloatComplex_ptr dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    dim3 threads( BLOCK_SIZE );
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    hipLaunchKernelGGL(clacpy_conj_kernel, dim3(blocks), dim3(threads), 0, queue->hip_stream() ,  n, dA1, lda1, dA2, lda2 );
}
