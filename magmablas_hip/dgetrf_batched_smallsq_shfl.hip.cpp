#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zgetrf_batched_smallsq_shfl.hip.cpp, normal z -> d, Fri Aug 25 13:17:09 2023
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.hip.hpp"
#include "shuffle.hip.hpp"
#include "batched_kernel_param.h"

// This kernel uses registers for matrix storage, shared mem. and shuffle for communication.
// It also uses lazy swap.
//HIP_DYNAMIC_SHARED( double, ddata)
template<int N, int NSHFL>
__global__
__launch_bounds__(NSHFL)
void
dgetrf_batched_smallsq_shfl_kernel( double** dA_array, int ldda,
                                magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount)
{
    HIP_DYNAMIC_SHARED( double, ddata)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    double* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];

    double rA[N]  = {MAGMA_D_ZERO};
    double  y[N]  = {MAGMA_D_ZERO};
    double reg    = MAGMA_D_ZERO;
    double update = MAGMA_D_ZERO;

    int max_id, current_piv_tx, rowid = tx, linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;
    // shared memory pointers
    double* sx = (double*)(ddata);
    int* sipiv = (int*)(sx + blockDim.y * NSHFL);
    sx += ty * NSHFL;
    sipiv += ty * (NSHFL+1);
    volatile int* scurrent_piv_tx = (volatile int*)(sipiv + NSHFL);

    // read
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        sx[ rowid ] = fabs(MAGMA_D_REAL( rA[i] )) + fabs(MAGMA_D_IMAG( rA[i] ));
        magmablas_syncwarp();
        rx_abs_max = sx[i];
        max_id = i;
        #pragma unroll
        for(int j = i; j < N; j++){
            if( sx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = sx[j];
            }
        }
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_D_ZERO : MAGMA_D_ONE;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            (*scurrent_piv_tx) = tx;
        }
        else if(rowid == i){
            rowid = max_id;
        }
        current_piv_tx = (*scurrent_piv_tx);
        magmablas_syncwarp();

        #pragma unroll
        for(int j = i; j < N; j++){
            y[j] = update * magmablas_dshfl( rA[j], current_piv_tx, NSHFL);
        }
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_D_ONE : MAGMA_D_DIV(MAGMA_D_ONE, y[i] );
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * y[j];
            }
        }
    }

    // write
    if( tx == 0 ){
        (*info) = (magma_int_t)linfo;
    }
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    dgetrf_batched_smallsq_noshfl computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges.
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_dgetrf_batched_smallsq_shfl(
    magma_int_t n,
    double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m = n;

    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0) return 0;

    const magma_int_t ntcol = magma_get_dgetrf_batched_ntcol(m, n);
    magma_int_t shmem  = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(double);
                shmem += ntcol * 1 * sizeof(int);
    dim3 threads(magma_ceilpow2(m), ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    switch(m){
        case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 1, magma_ceilpow2( 1)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 2, magma_ceilpow2( 2)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 3, magma_ceilpow2( 3)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 4, magma_ceilpow2( 4)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 5, magma_ceilpow2( 5)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 6, magma_ceilpow2( 6)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 7, magma_ceilpow2( 7)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 8, magma_ceilpow2( 8)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case  9: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel< 9, magma_ceilpow2( 9)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 10: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<10, magma_ceilpow2(10)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 11: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<11, magma_ceilpow2(11)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 12: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<12, magma_ceilpow2(12)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 13: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<13, magma_ceilpow2(13)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 14: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<14, magma_ceilpow2(14)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 15: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<15, magma_ceilpow2(15)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 16: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<16, magma_ceilpow2(16)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 17: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<17, magma_ceilpow2(17)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 18: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<18, magma_ceilpow2(18)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 19: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<19, magma_ceilpow2(19)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 20: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<20, magma_ceilpow2(20)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 21: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<21, magma_ceilpow2(21)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 22: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<22, magma_ceilpow2(22)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 23: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<23, magma_ceilpow2(23)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 24: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<24, magma_ceilpow2(24)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 25: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<25, magma_ceilpow2(25)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 26: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<26, magma_ceilpow2(26)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 27: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<27, magma_ceilpow2(27)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 28: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<28, magma_ceilpow2(28)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 29: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<29, magma_ceilpow2(29)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 30: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<30, magma_ceilpow2(30)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 31: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<31, magma_ceilpow2(31)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        case 32: hipLaunchKernelGGL(HIP_KERNEL_NAME(dgetrf_batched_smallsq_shfl_kernel<32, magma_ceilpow2(32)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
    }
    return arginfo;
}
