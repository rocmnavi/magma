#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah
*/

#ifndef TRMM_TEMPLATE_KERNEL_BATCHED_CUH
#define TRMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.hip.hpp"
#include "trmm_template_device.hip.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
void trmm_template_batched_lNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray,  int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB)
{
    int batchid = blockIdx.z;

    trmm_small_template_device_lNx<T, NB>(
            uplo, diag,
            m, n,
            alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
                   Barray[batchid] + coffB * lddb + roffB, lddb);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
void trmm_template_batched_lTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB)
{
    int batchid = blockIdx.z;

    trmm_small_template_device_lTx<T, NB, CONJA>(
            uplo, diag,
            m, n,
            alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
                   Barray[batchid] + coffB * lddb + roffB, lddb);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
static __global__
void trmm_template_batched_rNx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB)
{
    int batchid = blockIdx.z;

    trmm_small_template_device_rNx<T, NB>(
            uplo, diag,
            m, n,
            alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
                   Barray[batchid] + coffB * lddb + roffB, lddb);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
static __global__
void trmm_template_batched_rTx_kernel(
        magma_uplo_t uplo, magma_diag_t diag,
        int m, int n,
        T alpha, T** Aarray, int ldda,
                 T** Barray,  int lddb,
        int roffA, int coffA, int roffB, int coffB)
{
    int batchid = blockIdx.z;

    trmm_small_template_device_rTx<T, NB, CONJA>(
            uplo, diag,
            m, n,
            alpha, Aarray[batchid] + coffA * ldda + roffA, ldda,
                   Barray[batchid] + coffB * lddb + roffB, lddb);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// lNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_batched_lNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, NB ), 1, ibatch );

        hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_batched_lNx_kernel<T, NB>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA_array+i, ldda, dB_array+i, lddb,
        roffA, coffA, roffB, coffB);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// lTx, lCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_batched_lTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( n, NB ), 1, ibatch );

        hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_batched_lTx_kernel<T, NB, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA_array+i, ldda, dB_array+i, lddb,
        roffA, coffA, roffB, coffB);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rNx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB>
void trmm_template_batched_rNx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv( m, NB ), 1, ibatch );

        hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_batched_rNx_kernel<T, NB>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA_array+i, ldda, dB_array+i, lddb,
        roffA, coffA, roffB, coffB);
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// rTx, rCx
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int NB, const int CONJA>
void trmm_template_batched_rTx(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    T alpha, T** dA_array, magma_int_t ldda,
             T** dB_array, magma_int_t lddb,
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(NB, NB, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);

        dim3 grid( magma_ceildiv( m, NB ), 1, ibatch );
        hipLaunchKernelGGL(HIP_KERNEL_NAME(trmm_template_batched_rTx_kernel<T, NB, CONJA>), dim3(grid), dim3(threads), 0, queue->hip_stream() , uplo, diag, m, n, alpha, dA_array+i, ldda, dB_array+i, lddb,
        roffA, coffA, roffB, coffB);
    }
}
#endif //TRMM_TEMPLATE_KERNEL_BATCHED_CUH
