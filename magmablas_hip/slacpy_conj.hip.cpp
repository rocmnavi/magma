#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zlacpy_conj.hip.cpp, normal z -> s, Mon Jul 15 16:57:38 2024

*/
#include "magma_internal.h"

#define BLOCK_SIZE 64

/******************************************************************************/
// copy & conjugate a single vector of length n.
// TODO: this was modeled on the old sswap routine. Update to new slacpy code for 2D matrix?

__global__ void slacpy_conj_kernel(
    int n,
    float *A1, int lda1,
    float *A2, int lda2 )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int offset1 = x*lda1;
    int offset2 = x*lda2;
    if ( x < n )
    {
        A2[offset2] = MAGMA_S_CONJ( A1[offset1] );
    }
}


/******************************************************************************/
extern "C" void 
magmablas_slacpy_conj(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1, 
    magmaFloat_ptr dA2, magma_int_t lda2,
    magma_queue_t queue )
{
    dim3 threads( BLOCK_SIZE );
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    hipLaunchKernelGGL(slacpy_conj_kernel, dim3(blocks), dim3(threads), 0, queue->hip_stream() ,  n, dA1, lda1, dA2, lda2 );
}
