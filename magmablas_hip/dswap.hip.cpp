#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023
       
       @author Mark Gates

       @generated from magmablas_hip/zswap.hip.cpp, normal z -> d, Fri Aug 25 13:16:59 2023

*/
#include "magma_internal.h"

#define NB 64


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
__global__ void dswap_kernel(
    int n,
    double *x, int incx,
    double *y, int incy )
{
    double tmp;
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    if ( ind < n ) {
        x += ind*incx;
        y += ind*incy;
        tmp = *x;
        *x  = *y;
        *y  = tmp;
    }
}


/***************************************************************************//**
    Purpose:
    =============
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      DOUBLE PRECISION array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      DOUBLE PRECISION array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap
*******************************************************************************/
extern "C" void 
magmablas_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx, 
    magmaDouble_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( n, NB ) );
    hipLaunchKernelGGL(dswap_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  n, dx, incx, dy, incy );
}
