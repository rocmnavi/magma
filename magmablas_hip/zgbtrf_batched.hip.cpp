#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah
       @author Stan Tomov

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.hip.hpp"
#include "shuffle.hip.hpp"
#include "batched_kernel_param.h"

// use this so magmasubs will replace with relevant precision, so we can comment out       
// the switch case that causes compilation failure                                         
#define PRECISION_z

#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)             (max(1,64/M))
#endif

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap. 
template<int N, int NPOW2>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(NTCOL(N)*NPOW2)
#endif
void
zgbtrf_batched_kernel( magmaDoubleComplex** dA_array, int ldda,
                       magma_int_t** ipiv_array, magma_int_t *info_array, int batchCount)
{
    HIP_DYNAMIC_SHARED( magmaDoubleComplex, zdata)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dA_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    magma_int_t* info = &info_array[batchid];

    magmaDoubleComplex rA[N]  = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
    magmaDoubleComplex update = MAGMA_Z_ZERO;

    int max_id, rowid = tx;
    int linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(zdata);
    double* dsx = (double*)(sx + blockDim.y * NPOW2);
    int* sipiv = (int*)(dsx + blockDim.y * NPOW2);
    sx    += ty * NPOW2;
    dsx   += ty * NPOW2;
    sipiv += ty * NPOW2;

    // read
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        // izamax and find pivot  
        dsx[ rowid ] = fabs(MAGMA_Z_REAL( rA[i] )) + fabs(MAGMA_Z_IMAG( rA[i] ));
        magmablas_syncwarp();
        rx_abs_max = dsx[i];
        max_id = i;
        #pragma unroll
        for(int j = i+1; j < N; j++){
            if( dsx[j] > rx_abs_max){
                max_id = j;
                rx_abs_max = dsx[j];
            }
        }
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = i; j < N; j++){
                sx[j] = update * rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id;
        }
        magmablas_syncwarp();

        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] );
       
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        magmablas_syncwarp();
    }

    // write
    if( tx == 0 ){
        (*info) = (magma_int_t)linfo;
    }
    if(tx < N) {
        ipiv[ tx ] = (magma_int_t)(sipiv[tx] + 1);  // fortran indexing
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
// This is the non-pivoting version.
template<int N, int NPOW2>
__global__
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(NTCOL(N)*NPOW2)
#endif
void
zgbtrf_batched_np_kernel( magmaDoubleComplex** dA_array, int ldda,
                          magma_int_t *info_array, int batchCount)
{
    HIP_DYNAMIC_SHARED( magmaDoubleComplex, zdata)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA = dA_array[batchid];
    magma_int_t* info = &info_array[batchid];

    magmaDoubleComplex rA[N]  = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg    = MAGMA_Z_ZERO;
    
    int rowid = tx, linfo = 0;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(zdata);
    sx    += ty * NPOW2;

    // read
    if( tx < N ){
        #pragma unroll
        for(int i = 0; i < N; i++){
            rA[i] = dA[ i * ldda + tx ];
        }
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        rx_abs_max = fabs(MAGMA_Z_REAL( rA[i] )) + fabs(MAGMA_Z_IMAG( rA[i] ));
        magmablas_syncwarp();

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (i+1) : linfo;
        
        magmablas_syncwarp();
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] );

        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        magmablas_syncwarp();
    }

    // write
    if( tx == 0 ){
        (*info) = (magma_int_t)linfo;
    }
    if(tx < N) {
        #pragma unroll
        for(int i = 0; i < N; i++){
            dA[ i * ldda + rowid ] = rA[i];
        }
    }
}

/***************************************************************************//**
    Purpose
    -------
    zgbtrf_batched computes the LU factorization of a square N-by-N matrix A
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
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
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
magma_zgbtrf_batched(
    magma_int_t use_pivoting,
    magma_int_t m,
    magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    if ( use_pivoting !=0 && use_pivoting !=1 )
        arginfo = -1;
    if( m < 0 )
        arginfo = -2;
    else if ( n < 0 )
        arginfo = -3;
    else if ( kl < 0 )
        arginfo = -4;
    else if ( ku < 0 )
        arginfo = -5;
    else if ( ldda < kl + ku + 1 )
        arginfo = -7;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 ) return 0;

    #ifdef MAGMA_HAVE_HIP
    const magma_int_t ntcol = NTCOL(n);
    #else
    const magma_int_t ntcol = 1; //magma_get_zgetrf_batched_ntcol(m, n);
    #endif

    magma_int_t shmem  = ntcol * magma_ceilpow2(m) * sizeof(int);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(double);
                shmem += ntcol * magma_ceilpow2(m) * sizeof(magmaDoubleComplex); 
    dim3 threads(magma_ceilpow2(m), ntcol, 1);
    const magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);

    if (use_pivoting == 0)
        switch(m){
            /*
              case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 1, magma_ceilpow2( 1)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 2, magma_ceilpow2( 2)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 3, magma_ceilpow2( 3)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 4, magma_ceilpow2( 4)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 5, magma_ceilpow2( 5)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 6, magma_ceilpow2( 6)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 7, magma_ceilpow2( 7)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 8, magma_ceilpow2( 8)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case  9: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel< 9, magma_ceilpow2( 9)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 10: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<10, magma_ceilpow2(10)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 11: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<11, magma_ceilpow2(11)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 12: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<12, magma_ceilpow2(12)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 13: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<13, magma_ceilpow2(13)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 14: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<14, magma_ceilpow2(14)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 15: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<15, magma_ceilpow2(15)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 16: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<16, magma_ceilpow2(16)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 17: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<17, magma_ceilpow2(17)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 18: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<18, magma_ceilpow2(18)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 19: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<19, magma_ceilpow2(19)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 20: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<20, magma_ceilpow2(20)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 21: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<21, magma_ceilpow2(21)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 22: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<22, magma_ceilpow2(22)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 23: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<23, magma_ceilpow2(23)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 24: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<24, magma_ceilpow2(24)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 25: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<25, magma_ceilpow2(25)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 26: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<26, magma_ceilpow2(26)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 27: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<27, magma_ceilpow2(27)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 28: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<28, magma_ceilpow2(28)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 29: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<29, magma_ceilpow2(29)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 30: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<30, magma_ceilpow2(30)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
              case 31: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<31, magma_ceilpow2(31)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, ipiv_array, info_array, batchCount); break;
            */
        case 32: hipLaunchKernelGGL(HIP_KERNEL_NAME(zgbtrf_batched_np_kernel<32, magma_ceilpow2(32)>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), dA_array, ldda, info_array, batchCount); break;
        default: printf("error: size %lld is not supported\n", (long long) m);
        }
    else
        printf("error: pivoting is not supported yet\n");
    return arginfo;
}
