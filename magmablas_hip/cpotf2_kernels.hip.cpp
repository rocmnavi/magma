#include "hip/hip_runtime.h"
/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad

   @generated from magmablas_hip/zpotf2_kernels.hip.cpp, normal z -> c, Mon Oct 28 11:12:20 2024
 */
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_c

#if defined(VERSION31)
    #define ENABLE_COND1
    #define ENABLE_COND2
    #define ENABLE_COND4
    #define ENABLE_COND5
    #define ENABLE_COND6
#endif

#define MAX_NTCOL 8
#if defined(PRECISION_s)
#define NTCOL2   (4)
#define NTCOL1   (8)
#elif defined(PRECISION_d)
#define NTCOL2   (2)
#define NTCOL1   (4)
#else
#define NTCOL2   (1)
#define NTCOL1   (1)
#endif


#include "cpotf2_devicesfunc.hip.hpp"

#define A(i_, j_)  (dA + (i_) + (j_)*ldda)
/******************************************************************************/
__global__ void cpotf2_smlpin_fixwidth_kernel(int m, magmaFloatComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        if(threadIdx.x < m-i){
            cpotf2_smlpout_fixwidth_device(m-i, A(localstep+i, 0), A(localstep+i, localstep+i), ldda, localstep+i, gbstep, dinfo);
        }
    }
}
/******************************************************************************/
__global__ void cpotf2_smlpin_anywidth_kernel(int m, magmaFloatComplex *dA, int ldda, int localstep, int gbstep, magma_int_t *dinfo)
{
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        int ib = min(m-i, POTF2_NB);
        if(threadIdx.x < m-i){
            cpotf2_smlpout_anywidth_device(m-i, ib, A(localstep+i, 0), A(localstep+i, localstep+i), ldda, localstep+i, gbstep, dinfo);
        }
    }
}
/******************************************************************************/
__global__ void cpotf2_smlpin_fixwidth_kernel_batched(int m,
        magmaFloatComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    magmaFloatComplex *dA = dA_array[batchid] + aj * lda + ai;
    if (batchid >= batchCount) return;
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        //if(threadIdx.x < m-i){
            cpotf2_smlpout_fixwidth_device(m-i, dA+localstep+i, dA+localstep+i+(localstep+i)*lda, lda, localstep+i, gbstep, &(info_array[batchid]));
        //}
    }
}


/******************************************************************************/
__global__ void cpotf2_smlpin_anywidth_kernel_batched(int m,
        magmaFloatComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    magmaFloatComplex *dA = dA_array[batchid] + aj * lda + ai;
    if (batchid >= batchCount) return;
    #pragma unroll
    for(int i = 0; i < m; i+= POTF2_NB){
        int ib = min(m-i, POTF2_NB);
        //if(threadIdx.x < m-i){
            cpotf2_smlpout_anywidth_device(m-i, ib, dA+localstep+i, dA+localstep+i+(localstep+i)*lda, lda, localstep+i, gbstep, &(info_array[batchid]));
        //}
    }
}
/******************************************************************************/
__global__ void cpotf2_smlpout_fixwidth_kernel(int m,
        magmaFloatComplex *dA, int lda,
        int localstep, int gbstep, magma_int_t *dinfo)
{
    cpotf2_smlpout_fixwidth_device(m, dA+localstep, dA+localstep+localstep*lda, lda, localstep, gbstep, dinfo );
}


/******************************************************************************/
__global__ void cpotf2_smlpout_anywidth_kernel(int m, int n,
        magmaFloatComplex *dA, int lda,
        int localstep, int gbstep, magma_int_t *dinfo)
{
    cpotf2_smlpout_anywidth_device(m, n, dA+localstep, dA+localstep+localstep*lda, lda, localstep, gbstep, dinfo );
}



/******************************************************************************/
__global__ void cpotf2_smlpout_fixwidth_kernel_batched(int m,
        magmaFloatComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    magmaFloatComplex *dA = dA_array[batchid] + aj * lda + ai;
    cpotf2_smlpout_fixwidth_device(m, dA+localstep, dA+localstep+localstep*lda, lda, localstep, gbstep, &(info_array[batchid]));
}


/******************************************************************************/
__global__ void cpotf2_smlpout_anywidth_kernel_batched(int m, int n,
        magmaFloatComplex **dA_array, int ai, int aj, int lda,
        int localstep, int gbstep, magma_int_t *info_array, const int batchCount)
{
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if (batchid >= batchCount) return;
    magmaFloatComplex *dA = dA_array[batchid] + aj * lda + ai;
    cpotf2_smlpout_anywidth_device(m, n, dA+localstep, dA+localstep+localstep*lda, lda, localstep, gbstep, &(info_array[batchid]));
}

/******************************************************************************/
extern "C" magma_int_t
magma_cpotrf_lpout_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t roundup_m = m;
    // rounding up need more investigation since it coul dmodify the matrix out of its bound
    //magma_int_t m8  = magma_roundup( m, 8 );
    //magma_int_t roundup_m = m8 > lda ? m : m8;
    //magma_int_t m32 = magma_roundup( m, 32 );
    //magma_int_t roundup_m = m32 > lda ? m : m32;

    magma_int_t  ib, rows;

    for (magma_int_t j = 0; j < n; j += POTF2_NB) {
        ib   = min(POTF2_NB, n-j);
        rows = roundup_m-j;

        // tuning ntcol
        magma_int_t ntcol;  // for z precision, the best tuning is at NTCOL = 1 for all sizes
        if (rows > 64) ntcol = 1;
        else if (rows > 32) ntcol = NTCOL2;
        else ntcol = NTCOL1;
        // end of tuning ntcol

        const magma_int_t nTB = magma_ceildiv( batchCount, ntcol );
        dim3 dimGrid(nTB, 1, 1);
        magma_int_t nbth = rows;
        magma_int_t shared_mem_size = ntcol * (sizeof(magmaFloatComplex)*(nbth+POTF2_NB)*POTF2_NB);
        dim3 threads(nbth, ntcol);

        if (shared_mem_size > 47000)
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        if (ib == POTF2_NB) {
            hipLaunchKernelGGL(cpotf2_smlpout_fixwidth_kernel_batched, dim3(dimGrid), dim3(threads), shared_mem_size, queue->hip_stream() , rows, dA_array, ai, aj, lda, j, gbstep, info_array, batchCount);
        }
        else {
            hipLaunchKernelGGL(cpotf2_smlpout_anywidth_kernel_batched, dim3(dimGrid), dim3(threads), shared_mem_size, queue->hip_stream() , rows, ib, dA_array, ai, aj, lda, j, gbstep, info_array, batchCount);
        }
    }

    return arginfo;
}
/******************************************************************************/
extern "C" magma_int_t
magma_cpotrf_lpin_batched(
        magma_uplo_t uplo, magma_int_t n,
        magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }
    dim3 grid(batchCount, 1, 1);
    dim3 threads(n, 1, 1);
    magma_int_t shared_mem_size = sizeof(magmaFloatComplex) * (n+POTF2_NB)*POTF2_NB;
    if (shared_mem_size > 47000) {
        arginfo = -33;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }



    if( n % POTF2_NB == 0){
        hipLaunchKernelGGL(cpotf2_smlpin_fixwidth_kernel_batched, dim3(grid), dim3(threads), shared_mem_size, queue->hip_stream() , n, dA_array, ai, aj, lda, 0, gbstep, info_array, batchCount);
    }
    else{
        hipLaunchKernelGGL(cpotf2_smlpin_anywidth_kernel_batched, dim3(grid), dim3(threads), shared_mem_size, queue->hip_stream() , n, dA_array, ai, aj, lda, 0, gbstep, info_array, batchCount);
    }

    return arginfo;
}


/******************************************************************************/
extern "C" magma_int_t
magma_cpotf2_lpout(
        magma_uplo_t uplo, magma_int_t n,
        magmaFloatComplex *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t m = n;
    magma_int_t arginfo = 0;

    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (m < 0 || n < 0 ) {
        arginfo = -2;
    } else if (lda < max(1,m)) {
        arginfo = -4;
    } else if (m < n) {
        arginfo = -10;
    }
    if (uplo == MagmaUpper) {
        fprintf( stderr, "%s: uplo=upper is not yet implemented\n", __func__ );
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t roundup_m = m;
    // rounding up need more investigation since it coul dmodify the matrix out of its bound
    //magma_int_t m8  = magma_roundup( m, 8 );
    //magma_int_t roundup_m = m8 > lda ? m : m8;
    //magma_int_t m32 = magma_roundup( m, 32 );
    //magma_int_t roundup_m = m32 > lda ? m : m32;

    magma_int_t  ib, rows;

    for (magma_int_t j = 0; j < n; j += POTF2_NB) {
        ib   = min(POTF2_NB, n-j);
        rows = roundup_m-j;

        dim3 dimGrid(1, 1, 1);
        magma_int_t nbth = rows;
        magma_int_t shared_mem_size = sizeof(magmaFloatComplex)*(nbth+POTF2_NB)*POTF2_NB;
        dim3 threads(nbth, 1, 1);

        if (shared_mem_size > 47000)
        {
            arginfo = -33;
            magma_xerbla( __func__, -(arginfo) );
            return arginfo;
        }

        if (ib == POTF2_NB)
        {
            hipLaunchKernelGGL(cpotf2_smlpout_fixwidth_kernel, dim3(dimGrid), dim3(threads), shared_mem_size, queue->hip_stream() , rows, dA, lda, j, gbstep, dinfo );
        } else {
            hipLaunchKernelGGL(cpotf2_smlpout_anywidth_kernel, dim3(dimGrid), dim3(threads), shared_mem_size, queue->hip_stream() , rows, ib, dA, lda, j, gbstep, dinfo );
        }
    }

    return arginfo;
}

/******************************************************************************/
extern "C" magma_int_t
magma_cpotf2_lpin(
        magma_uplo_t uplo, magma_int_t n,
        magmaFloatComplex *dA, magma_int_t ldda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    // Quick return if possible
    if ( n == 0 ) {
        return arginfo;
    }
    dim3 grid(1, 1, 1);
    dim3 threads(n, 1, 1);
    magma_int_t shared_mem_size = sizeof(magmaFloatComplex) * (n+POTF2_NB)*POTF2_NB;
    if (shared_mem_size > 47000) {
        arginfo = -33;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n % POTF2_NB == 0){
        hipLaunchKernelGGL(cpotf2_smlpin_fixwidth_kernel, dim3(grid), dim3(threads), shared_mem_size, queue->hip_stream() , n, dA, ldda, 0, gbstep, dinfo);
    }
    else{
        hipLaunchKernelGGL(cpotf2_smlpin_anywidth_kernel, dim3(grid), dim3(threads), shared_mem_size, queue->hip_stream() , n, dA, ldda, 0, gbstep, dinfo);
    }
    return arginfo;
}
