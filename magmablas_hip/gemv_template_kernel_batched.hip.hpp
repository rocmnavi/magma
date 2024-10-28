#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#ifndef GEMV_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMV_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.hip.hpp" // use make_FloatingPoint
#include "gemv_template_device.hip.hpp"


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
gemvn_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int lda, int strideA,
    T const * const * x_array, T const * x, int incx, int stridex,
    T beta, T**  y_array, T* y, int incy, int stridey )
{
    const int batchid = blockIdx.z;
    const T* dA = (A_array == NULL) ? (A + batchid * strideA) : A_array[batchid];
    const T* dx = (x_array == NULL) ? (x + batchid * stridex) : x_array[batchid];
    T*       dy = (y_array == NULL) ? (y + batchid * stridey) : y_array[batchid];

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>
    (m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvn_template_batched(
    magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex,
    T beta, T** dy_array, T* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(m, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA,
                       dx_array_i, dx+(i*stridex), incx, stridex,
                beta,  dy_array_i, dy+(i*stridey), incy, stridey );
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
__global__ void
gemvc_kernel_batched(
    int m, int n, T alpha,
    T const * const * A_array, T const * A, int lda,  int strideA,
    T const * const * x_array, T const * x, int incx, int stridex,
    T beta, T**  y_array, T* y, int incy, int stridey )
{
    int batchid = blockIdx.z;
    const T* dA = (A_array == NULL) ? (A + batchid * strideA) : A_array[batchid];
    const T* dx = (x_array == NULL) ? (x + batchid * stridex) : x_array[batchid];
    T*       dy = (y_array == NULL) ? (y + batchid * stridey) : y_array[batchid];

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>
    (m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_template_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, T alpha,
    T const * const * dA_array, T const * dA, magma_int_t ldda, magma_int_t strideA,
    T const * const * dx_array, T const * dx, magma_int_t incx, magma_int_t stridex,
    T beta, T** dy_array, T* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(n, TILE_SIZE), 1, ibatch );

        T const * const * dA_array_i = (dA_array == NULL) ? NULL : dA_array+i;
        T const * const * dx_array_i = (dx_array == NULL) ? NULL : dx_array+i;
        T**               dy_array_i = (dy_array == NULL) ? NULL : dy_array+i;

        if (trans == MagmaConjTrans) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA,
                           dx_array_i, dx+(i*stridex), incx, stridex,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey );
        }
        else if (trans == MagmaTrans) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvc_kernel_batched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, alpha, dA_array_i, dA+(i*strideA), ldda, strideA,
                           dx_array_i, dx+(i*stridex), incx, stridex,
                    beta,  dy_array_i, dy+(i*stridey), incy, stridey );
        }
    }
}

#endif
