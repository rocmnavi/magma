#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023
       
       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_KERNEL_CUH
#define TRMM_TEMPLATE_KERNEL_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.hip.hpp"
#include "trmm_template_device.hip.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB * NB * 1)
void trmm_template_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    trmm_small_template_device_lNx<T, NB>
    (uplo, diag, m, n, alpha, A, ldda, B, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB * NB * 1)
void trmm_template_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    trmm_small_template_device_lTx<T, NB, CONJA>
    (uplo, diag, m, n, alpha, A, ldda, B, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
__launch_bounds__(NB * NB * 1)
void trmm_template_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    trmm_small_template_device_rNx<T, NB>
    (uplo, diag, m, n, alpha, A, ldda, B, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
__launch_bounds__(NB * NB * 1)
void trmm_template_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag, 
        int m, int n, 
        T alpha, T* A, int ldda, 
                 T* B, int lddb)
{
    trmm_small_template_device_rTx<T, NB, CONJA>
    (uplo, diag, m, n, alpha, A, ldda, B, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_lNx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    dim3 threads(NB, NB, 1);
    dim3 grid( magma_ceildiv( n, NB ), 1, 1 );
    hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_lNx_kernel<T, NB>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA, ldda, dB, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_lTx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    dim3 threads(NB, NB, 1);
    dim3 grid( magma_ceildiv( n, NB ), 1, 1 );
    hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_lTx_kernel<T, NB, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA, ldda, dB, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_rNx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    dim3 threads(NB, NB, 1);
    dim3 grid( magma_ceildiv( m, NB ), 1, 1 );
    hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_rNx_kernel<T, NB>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA, ldda, dB, lddb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_rTx(
    magma_uplo_t uplo, magma_diag_t diag, 
    magma_int_t m, magma_int_t n, 
    T alpha, T* dA, magma_int_t ldda,
             T* dB, magma_int_t lddb,
    magma_queue_t queue)
{
    dim3 threads(NB, NB, 1);
    dim3 grid( magma_ceildiv( m, NB ), 1, 1 );
    hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_rTx_kernel<T, NB, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA, ldda, dB, lddb);
}

#endif //TRMM_TEMPLATE_KERNEL_CUH
