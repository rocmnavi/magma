#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas_hip/zcaxpycp.hip.cpp, mixed zc -> ds, Fri Aug 25 13:16:50 2023

*/
#include "magma_internal.h"

#define NB 64


/******************************************************************************/
// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__global__ void
dsaxpycp_kernel(
    int m,
    float *r,
    double *x,
    const double *b,
    double *w )
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_D_ADD( x[i], MAGMA_D_MAKE( MAGMA_S_REAL( r[i] ),
                                                MAGMA_S_IMAG( r[i] ) ) );
        w[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r (including conversion to double)  --and--
    copies w = b
*******************************************************************************/
extern "C" void
magmablas_dsaxpycp(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr w,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    hipLaunchKernelGGL(dsaxpycp_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, r, x, b, w );
}
