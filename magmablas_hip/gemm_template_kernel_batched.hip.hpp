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

#ifndef GEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMM_TEMPLATE_KERNEL_BATCHED_CUH

#include "gemm_template_device_defs.hip.hpp"
#include "gemm_template_device.hip.hpp"


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    extern __shared__ T* sdata_nn[];
    const int batchid = blockIdx.z;

    const int slda = BLK_M+1;    // +1 only required if A is transposed
    const int sldb = BLK_K+1;    // +1 always required
    T* sA = (T*)sdata_nn;        // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K;   // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K,
          Aarray[batchid] + LDA *  coffA + roffA, LDA,
          Barray[batchid] + LDB *  coffB + roffB, LDB,
          Carray[batchid] + LDC *  coffC + roffC, LDC,
          alpha, beta,
          sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    extern __shared__ T* sdata_nt[];
    const int batchid = blockIdx.z;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_nt;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_nt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K,
          Aarray[batchid] + LDA *  coffA + roffA, LDA,
          Barray[batchid] + LDB *  coffB + roffB, LDB,
          Carray[batchid] + LDC *  coffC + roffC, LDC,
          alpha, beta,
          sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    extern __shared__ T* sdata_tn[];
    const int batchid = blockIdx.z;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_tn;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K,
          Aarray[batchid] + LDA *  coffA + roffA, LDA,
          Barray[batchid] + LDB *  coffB + roffB, LDB,
          Carray[batchid] + LDC *  coffC + roffC, LDC,
          alpha, beta,
          sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int roffA, int coffA,
    int roffB, int coffB,
    int roffC, int coffC )
{
    extern __shared__ T* sdata_tt[];
    const int batchid = blockIdx.z;

    const int slda = BLK_M+1;  // +1 only required if A is transposed
    const int sldb = BLK_K+1;  // +1 always required
    T* sA = (T*)sdata_tt;      // sA is (BLK_M+1) x (BLK_K)
    T* sB = sA + slda * BLK_K; // sB is (BLK_K+1) x (BLK_N)

    gemm_template_device_tt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K,
          Aarray[batchid] + LDA *  coffA + roffA, LDA,
          Barray[batchid] + LDB *  coffB + roffB, LDB,
          Carray[batchid] + LDC *  coffC + roffC, LDC,
          alpha, beta,
          sA, slda, sB, sldb, NULL, 0 );
}


/******************************************************************************/
// kernel wrappers
// NN
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_batched_nn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), ibatch );
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_template_batched_nn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, queue->hip_stream() , m, n, k, dA_array+i, ldda, dB_array+i, lddb, dC_array+i, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// NT, NC
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_batched_nt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), ibatch );
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_template_batched_nt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, queue->hip_stream() , m, n, k, dA_array+i, ldda, dB_array+i, lddb, dC_array+i, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// TN, CN
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_batched_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), ibatch );
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_template_batched_tn_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, queue->hip_stream() , m, n, k, dA_array+i, ldda, dB_array+i, lddb, dC_array+i, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}


/******************************************************************************/
// TT, TC, CT, CC
template <typename T, const int DIM_X, const int DIM_Y,
         const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB,
         const int CONJA, const int CONJB>
void gemm_template_batched_tt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t roffA, magma_int_t coffA,
    magma_int_t roffB, magma_int_t coffB,
    magma_int_t roffC, magma_int_t coffC,
    magma_int_t batchCount, magma_queue_t queue)
{
    size_t shmem = 0;
    magma_int_t max_batchCount = queue->get_maxBatch();
    shmem += (BLK_M+1) * BLK_K * sizeof(T);  // sA
    shmem += (BLK_K+1) * BLK_N * sizeof(T);  // sB
    dim3 dimBlock(DIM_X, DIM_Y);
    for(magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), ibatch );
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_template_batched_tt_kernel<T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>), dim3(dimGrid), dim3(dimBlock), shmem, queue->hip_stream() , m, n, k, dA_array+i, ldda, dB_array+i, lddb, dC_array+i, lddc, alpha, beta, roffA, coffA, roffB, coffB, roffC, coffC);
    }
}

#endif //GEMM_TEMPLATE_KERNEL_BATCHED_CUH
