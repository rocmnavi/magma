#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023
       
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#ifndef HEMM_TEMPLATE_KERNEL_CUH
#define HEMM_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.hip.hpp"
#include "hemm_template_device.hip.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_ll_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta)
{
    hemm_template_device_ll
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, A, LDA, B, LDB, C, LDC, alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_lu_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta)
{
    hemm_template_device_lu
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, A, LDA, B, LDB, C, LDC, alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_rl_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta)
{
    hemm_template_device_rl
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, A, LDA, B, LDB, C, LDC, alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
static __global__
void hemm_template_ru_kernel(
    int M, int N, 
    const T* A, int LDA,
    const T* B, int LDB,
          T* C, int LDC,
    T alpha, T beta)
{
    hemm_template_device_ru
    <T, DIM, BLK_M, BLK_N, (BLK_M/DIM), (BLK_N/DIM), CONJA>
    ( M, N, A, LDA, B, LDB, C, LDC, alpha, beta );
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM, const int BLK_M, const int BLK_N, const int CONJA>
void hemm_template(
    magma_side_t side, magma_uplo_t uplo, 
    magma_int_t m, magma_int_t n, 
    const T* dA, magma_int_t ldda,
    const T* dB, magma_int_t lddb,
          T* dC, magma_int_t lddc,
    T alpha, T beta, magma_queue_t queue)
{
    dim3 threads(DIM, DIM, 1);
    dim3 grid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), 1 );
    if( side == MagmaLeft ){
        if(uplo == MagmaLower){
            hipLaunchKernelGGL(HIP_KERNEL_NAME(hemm_template_ll_kernel<T, DIM, BLK_M, BLK_N, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta );
        }else{
            hipLaunchKernelGGL(HIP_KERNEL_NAME(hemm_template_lu_kernel<T, DIM, BLK_M, BLK_N, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta );
        }
    }else{
        if(uplo == MagmaLower){
            hipLaunchKernelGGL(HIP_KERNEL_NAME(hemm_template_rl_kernel<T, DIM, BLK_M, BLK_N, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta );
        }else{
            hipLaunchKernelGGL(HIP_KERNEL_NAME(hemm_template_ru_kernel<T, DIM, BLK_M, BLK_N, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta );
        }
    }
}
#endif //HEMM_TEMPLATE_KERNEL_CUH
