/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas/zaxpycp.cu, normal z -> s, Fri Aug 25 13:18:31 2023

*/
#include "magma_internal.h"

#define NB 64

/******************************************************************************/
// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
saxpycp_kernel(
    int m,
    float *r,
    float *x,
    const float *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_S_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r  --and--
    copies r = b
*******************************************************************************/
extern "C" void
magmablas_saxpycp(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaFloat_ptr x,
    magmaFloat_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    saxpycp_kernel <<< grid, threads, 0, queue->cuda_stream() >>> ( m, r, x, b );
}
