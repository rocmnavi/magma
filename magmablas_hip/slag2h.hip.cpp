#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah
*/

#include <hip/hip_runtime.h>    // for CUDA_VERSION
#include "magma_internal.h"

#if defined(MAGMA_HAVE_HIP)
#include <hip/hip_fp16.h>
#endif

#define BLK_X 32
#define BLK_Y 4
#define MAX_BATCH    65000

// TODO: static is not working for HIP; removed from cuda as well
__device__ magma_int_t magma_flag = 0;
__device__ magma_int_t magma_flag_array[ MAX_BATCH ] = { 0 };
/******************************************************************************/
__device__
void slag2h_device(
    int m, int n,
    const float *A, int lda,
    magmaHalf *HA,  int ldha,
    float rmax, magma_int_t* dinfo)
{
#if CUDA_VERSION >= 7500 || defined(MAGMA_HAVE_HIP)
    const int gtx = blockIdx.x * BLK_X + threadIdx.x;
    const int gty = blockIdx.y * BLK_Y + threadIdx.y;

    float tmp;
    float neg_rmax = - rmax;

    for(int j = gty; j < n; j += gridDim.y * BLK_Y) {
        for(int i = gtx; i < m; i+= gridDim.x * BLK_X){
            tmp = A[j * lda + i];
            if ( (MAGMA_S_REAL(tmp) < neg_rmax) || (MAGMA_S_REAL(tmp) > rmax) ) {
                *dinfo  = 1;
            }
            HA[j * ldha + i] = __float2half( tmp );
        }
    }
#endif
}


/******************************************************************************/
__global__
__launch_bounds__(BLK_X*BLK_Y)
void slag2h_kernel(
        int m, int n,
        float const *dA, int lda,
        magmaHalf* dHA, int ldha,
        float rmax, magma_int_t* dinfo )
{
    slag2h_device(m, n, dA, lda, dHA, ldha, rmax, dinfo);
}


/******************************************************************************/
__global__
__launch_bounds__(BLK_X*BLK_Y)
void slag2h_kernel_batched(
        int m, int n,
        float const * const * dAarray, int lda,
        magmaHalf** dHAarray, int ldha,
        float rmax, magma_int_t* dinfo_array,
        magma_queue_t queue )
{
    const int batchid = blockIdx.z;
    slag2h_device( m, n, dAarray[batchid], lda, dHAarray[batchid], ldha, rmax, &dinfo_array[batchid]);
}

/******************************************************************************/
extern "C" void
magmablas_slag2h(
    magma_int_t m, magma_int_t n,
    float const * dA, magma_int_t lda,
    magmaHalf* dHA, magma_int_t ldha,
    magma_int_t *info, magma_queue_t queue)
{
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,m) )
        *info = -4;
    else if ( ldha < max(1,m) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    hipMemcpyToSymbol(HIP_SYMBOL(magma_flag), info, sizeof(magma_flag) );    // magma_flag = 0

    // there is no lapackf77_hlamch, please visit:
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    dim3 threads( BLK_X, BLK_Y );
    dim3 grid( magma_ceildiv(m, BLK_X), min(65000, magma_ceildiv(n, BLK_Y)), 1);

    hipLaunchKernelGGL(slag2h_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dA, lda, dHA, ldha, rmax, &magma_flag );

    hipMemcpyFromSymbol(info, HIP_SYMBOL(magma_flag), sizeof(magma_flag) );  // info = magma_flag

}


/******************************************************************************/
extern "C" void
magmablas_slag2h_batched(
    magma_int_t m, magma_int_t n,
    float const * const * dAarray, magma_int_t lda,
    magmaHalf** dHAarray, magma_int_t ldha,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if ( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( lda < max(1,m) )
        arginfo = -4;
    else if ( ldha < max(1,m) )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    memset( info_array, 0, batchCount * sizeof(magma_int_t) );    // init info_array to zero

    // there is no lapackf77_hlamch, please visit:
    // https://blogs.mathworks.com/cleve/2017/05/08/half-precision-16-bit-floating-point-arithmetic/
    float rmax = (float)(65504);

    dim3 threads( BLK_X, BLK_Y );
    const int maxBatch = MAX_BATCH;
    for(int i = 0; i < batchCount; i+=maxBatch){
        magma_int_t batch = min(maxBatch, batchCount-i);
        hipMemcpyToSymbol(HIP_SYMBOL(magma_flag_array), info_array + i, sizeof(magma_int_t) );

        dim3 grid( magma_ceildiv(m, BLK_X), magma_ceildiv(n, BLK_Y), batch);
        hipLaunchKernelGGL(slag2h_kernel_batched, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dAarray + i, lda, dHAarray + i, ldha, rmax, magma_flag_array, queue);

        hipMemcpyFromSymbol(info_array + i, HIP_SYMBOL(magma_flag_array), sizeof(magma_int_t) );
    }
}
