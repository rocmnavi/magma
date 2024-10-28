#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas_hip/zgeqrf_batched_smallsq.hip.cpp, normal z -> d, Mon Oct 28 11:12:19 2024
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.hip.hpp"
#include "batched_kernel_param.h"

#ifdef MAGMA_HAVE_HIP
#define block_sync    __syncthreads
#else
#define block_sync    __syncthreads
#endif

#if defined(MAGMA_HAVE_HIP)
#define DGEQRF_BATCH_SQ1D_MAX_THREADS    (64)

#elif defined(MAGMA_HAVE_CUDA)
#define DGEQRF_BATCH_SQ1D_MAX_THREADS    (128)

#endif

#define SLDA(N)    ( (N==15||N==23||N==31)? (N+2) : (N+1) )
template<int N>
__global__
__launch_bounds__(DGEQRF_BATCH_SQ1D_MAX_THREADS)
void
dgeqrf_batched_sq1d_reg_kernel(
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount)
{
    HIP_DYNAMIC_SHARED( double, zdata)
    constexpr int mtx_per_tb = (DGEQRF_BATCH_SQ1D_MAX_THREADS/N);
    const int tx = threadIdx.x % N;
    const int ty = threadIdx.x / N;
    const int batchid = blockIdx.x * mtx_per_tb + ty;
    if(batchid >= batchCount) return;
    if(ty      >= mtx_per_tb) return;

    const int slda  = SLDA(N);
    double* dA   = dA_array[batchid] + Aj * ldda + Ai;
    double* dtau = dtau_array[batchid] + taui;
    magma_int_t* info = &info_array[batchid];
    // shared memory pointers
    double* sA = (double*)(zdata + ty * slda * N);
    double* sdw = (double*)(zdata + mtx_per_tb * slda * N);
    sdw += ty * N;

    double rA[N] = {MAGMA_D_ZERO};
    double alpha, tau, tmp, zsum, scale = MAGMA_D_ZERO;
    double norm_no_alpha = MAGMA_D_ZERO, norm = MAGMA_D_ZERO, beta;

    if( tx == 0 ){
        (*info) = 0;
    }

    // init tau
    dtau[tx] = MAGMA_D_ZERO;
    // read
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }

    #pragma unroll
    for(int i = 0; i < N; i++){
        sA[ i * slda + tx] = rA[i];
        sdw[tx] = ( MAGMA_D_REAL(rA[i]) * MAGMA_D_REAL(rA[i]) + MAGMA_D_IMAG(rA[i]) * MAGMA_D_IMAG(rA[i]) );
        block_sync();
        alpha = sA[i * slda + i];
        norm_no_alpha = MAGMA_D_ZERO;
        #pragma unroll
        for(int j = i+1; j < N; j++){
            norm_no_alpha += sdw[j];
        }
        norm = norm_no_alpha + sdw[i];
        bool zero_nrm = (norm_no_alpha == 0) && (MAGMA_D_IMAG(alpha) == 0);

        norm = sqrt(norm);
        tmp   = rA[i];
        tau   = MAGMA_D_ZERO;
        scale = MAGMA_D_ONE;

        if(!zero_nrm) {
            beta = -copysign(norm, real(alpha));
            scale = (tx > i) ? MAGMA_D_DIV( MAGMA_D_ONE,  alpha - MAGMA_D_MAKE(beta, 0)) : scale;
            tau   = MAGMA_D_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );
            rA[i] *= scale;
            tmp = (tx == i)? MAGMA_D_MAKE(beta, MAGMA_D_ZERO) : rA[i];
            if(tx == i){
                dtau[i] = tau;
                rA[i]   = MAGMA_D_ONE;
            }
        }

        dA[ i * ldda + tx ] = tmp;
        rA[i] = (tx == i)   ? MAGMA_D_ONE  : rA[i];
        rA[i] = (tx < i )   ? MAGMA_D_ZERO : rA[i];
        tmp = MAGMA_D_CONJ( rA[i] ) * MAGMA_D_CONJ( tau );

        block_sync();
        #pragma unroll
        for(int j = i+1; j < N; j++){
            sA[j * slda + tx] = rA[j] * tmp;
        }
        block_sync();

        zsum = MAGMA_D_ZERO;
        #pragma unroll
        for(int j = i; j < N; j++){
            zsum += sA[tx * slda + j];
        }
        sA[tx * slda + N] = zsum;
        block_sync();

        #pragma unroll
        for(int j = i+1; j < N; j++){
            rA[j] -= rA[i] * sA[j * slda + N];
        }
        block_sync();
    }
}

/***************************************************************************//**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.

    This is a batched version of the routine, and works only for small
    square matrices of size up to 32.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit, the elements on and above the diagonal of the array
             contain the min(M,N)-by-N upper trapezoidal matrix R (R is
             upper triangular if m >= n); the elements below the diagonal,
             with the array TAU, represent the orthogonal matrix Q as a
             product of min(m,n) elementary reflectors (see Further
             Details).

    @param[in]
    ldda     INTEGER
             The leading dimension of the array dA.  LDDA >= max(1,M).
             To benefit from coalescent memory accesses LDDA must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_dgeqrf_batched_smallsq(
    magma_int_t n,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t batchCount, magma_queue_t queue )
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

    if( m == 0 || n == 0) return 0;

    magma_int_t ntcol = max(1, DGEQRF_BATCH_SQ1D_MAX_THREADS/n); //magma_get_dgeqrf_batched_ntcol(m, n);
    magma_int_t shmem = ( SLDA(m) * m * sizeof(double) );
    shmem            += ( m * sizeof(double) );
    shmem            *= ntcol;
    magma_int_t nth   = m;//magma_ceilpow2(m);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads(nth * ntcol, 1, 1);

    void *kernel_args[] = {&dA_array, &Ai, &Aj, &ldda, &dtau_array, &taui, &info_array, &batchCount};

    hipError_t e = hipSuccess;
    switch(m){
        case  1: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 1>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  2: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 2>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  3: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 3>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  4: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 4>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  5: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 5>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  6: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 6>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  7: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 7>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  8: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 8>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case  9: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel< 9>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 10: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<10>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 11: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<11>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 12: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<12>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 13: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<13>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 14: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<14>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 15: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<15>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 16: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<16>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 17: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<17>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 18: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<18>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 19: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<19>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 20: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<20>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 21: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<21>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 22: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<22>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 23: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<23>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 24: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<24>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 25: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<25>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 26: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<26>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 27: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<27>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 28: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<28>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 29: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<29>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 30: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<30>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 31: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<31>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        case 32: e = hipLaunchKernel((void*)dgeqrf_batched_sq1d_reg_kernel<32>, grid, threads, kernel_args, shmem, queue->hip_stream()); break;
        default: arginfo = -100;
    }

    if( e != hipSuccess ) {
        arginfo = -100;
    }

    return arginfo;
}
