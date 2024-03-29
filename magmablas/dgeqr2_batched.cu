/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magmablas/zgeqr2_batched.cu, normal z -> d, Fri Aug 25 13:18:48 2023
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define BLOCK_SIZE 256


#define dA(a_1,a_2) (dA  + (a_1) + (a_2)*(local_lda))


#include "dlarfg_devicesfunc.cuh"

/******************************************************************************/
static __device__
void dlarfx_device(
    int m, int n,  double *v, double *tau,
    double *dc, magma_int_t ldc, double* sum)
{
    if (n <= 0) return;
    if (MAGMA_D_EQUAL(*tau, MAGMA_D_ZERO) )  return; // check singularity

    const int tx = threadIdx.x;

    double lsum;

    for (int k=0; k < n; k++)
    {
        /* perform  w := v' * C  */
        if (tx < BLOCK_SIZE)
        {
            if (tx == 0)
                lsum = dc[0+ldc*k]; //since V[0] should be one
            else
                lsum = MAGMA_D_ZERO;
            for (int j = tx+1; j < m; j += BLOCK_SIZE) {
                lsum += MAGMA_D_MUL( MAGMA_D_CONJ( v[j] ), dc[j+ldc*k] );
            }

            sum[tx] = lsum;
        }

        magma_sum_reduce< BLOCK_SIZE >( tx, sum );
        __syncthreads();

        double z__1 = - MAGMA_D_CONJ(*tau) * sum[0];
        /*  C := C - v * w  */
        if (tx < BLOCK_SIZE)
        {
            for (int j = tx+1; j < m; j += BLOCK_SIZE)
                dc[j+ldc*k] += z__1 * v[j];
        }
        if (tx == 0) dc[0+ldc*k] += z__1;

        __syncthreads();
    }
}


/******************************************************************************/
static __device__
void dgeqr2_device( magma_int_t m, magma_int_t n,
                               double* dA, magma_int_t lda,
                               double *dtau,
                               double *dv,
                               double *sum,
                               double *swork,
                               double *scale,
                               double *sscale)
{
    //lapack dlarfg, compute the norm, scale and generate the householder vector
    dlarfg_device(m, dv, &(dv[1]), 1, dtau, swork, sscale, scale);

    __syncthreads();

    //update the trailing matix with the householder
    dlarfx_device(m, n, dv, dtau, dA, lda, sum);

    __syncthreads();
}


/******************************************************************************/


/******************************************************************************/
__global__
void dgeqr2_sm_kernel_batched(
        int m, int n,
        double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        double **dtau_array, magma_int_t taui )
{
    extern __shared__ double shared_data[];

    double* dA   = dA_array[blockIdx.z];
    double* dtau = dtau_array[blockIdx.z];

    dA   += Aj * lda + Ai;
    dtau += taui;

    double *sdata = (double*)shared_data;

    const int tx = threadIdx.x;

    __shared__ double scale;
    __shared__ double sum[ BLOCK_SIZE ];

    __shared__ double swork[ BLOCK_SIZE ];
    __shared__ double sscale;

    //load data from global to shared memory
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            sdata[j + s * m] = dA[j + s * lda];
        }
    }


    __syncthreads();

    for (int s=0; s < min(m,n); s++)
    {
        dgeqr2_device( m-s, n-(s+1),
                       &(sdata[s+(s+1)*m]), m,
                       dtau+s,
                       &(sdata[s+s*m]),
                       sum,
                       swork,
                       &scale,
                       &sscale);
    } // end of s

    //copy back to global memory
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            dA[j + s * lda] = sdata[j + s * m];
        }
    }
}


/******************************************************************************/
__global__
void dgeqr2_column_sm_kernel_batched(
        int m, int n,
        double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        double **dtau_array, magma_int_t taui )
{
    extern __shared__ double shared_data[];

    double* dA   = dA_array[blockIdx.z];
    double* dtau = dtau_array[blockIdx.z];
    dA   += Aj * lda + Ai;
    dtau += taui;

    double *sdata = (double*)shared_data;

    __shared__ double scale;
    __shared__ double sum[ BLOCK_SIZE ];

    __shared__ double swork[ BLOCK_SIZE ];
    __shared__ double sscale;

    const int tx = threadIdx.x;

    for (int s=0; s < min(m,n); s++)
    {
        //load one vector in shared memory: sdata
        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            sdata[j] = dA[s + j + s * lda];
        }

        __syncthreads();

        //sdata is written
        dgeqr2_device(m-s, n-(s+1),
                                &(dA[s+(s+1)*lda]), lda,
                                dtau+s,
                                sdata,
                                sum,
                                swork,
                                &scale,
                                &sscale);

        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            dA[s + j + s * lda] = sdata[j];
        }

        __syncthreads();
    }
}


/******************************************************************************/
__global__
void dgeqr2_kernel_batched(
        int m, int n,
        double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        double **dtau_array, magma_int_t taui )
{
    double* dA   = dA_array[blockIdx.z];
    double* dtau = dtau_array[blockIdx.z];
    dA   += Aj * lda + Ai;
    dtau += taui;

    __shared__ double scale;
    __shared__ double sum[ BLOCK_SIZE ];

    __shared__ double swork[ BLOCK_SIZE ];
    __shared__ double sscale;



    for (int s=0; s < min(m,n); s++)
    {
        dgeqr2_device( m-s, n-(s+1),
                       &(dA[s+(s+1)*lda]), lda,
                       dtau+s,
                       &(dA[s+s*lda]),
                       sum,
                       swork,
                       &scale,
                       &sscale );
    }
}


/******************************************************************************/
extern "C" magma_int_t
magma_dgeqr2_fused_batched(
        magma_int_t m, magma_int_t n,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        double **dtau_array, magma_int_t taui,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t arginfo = 0;

    if (m < 0)
        arginfo = -1;
    else if (n < 0 || n > 32)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // try the register version
    arginfo = magma_dgeqr2_fused_reg_batched(
                m, n, dA_array, Ai, Aj, ldda,
                dtau_array, taui, info_array, 0, batchCount, queue );
    if ( arginfo == 0 ) return arginfo;

    // register version did not launch kernel
    // try shared memory version
    magma_int_t nthreads = magma_get_dgeqr2_fused_sm_batched_nthreads(m, n);
    arginfo = magma_dgeqr2_fused_sm_batched(
                m, n, dA_array, Ai, Aj, ldda,
                dtau_array, taui, info_array, nthreads, 0, batchCount, queue );

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    DGEQR2 computes a QR factorization of a real m by n matrix A:
    A = Q * R.

    This version implements the right-looking QR with non-blocking.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

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
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

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

    @ingroup magma_geqr2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_dgeqr2_batched(magma_int_t m, magma_int_t n,
                     double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
                     double **dtau_array, magma_int_t taui,
                     magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t k = min(m,n);

    // first, try the fused geqr2
    arginfo = magma_dgeqr2_fused_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue);
    if ( arginfo == 0 ) return arginfo;

    // reaching this point means that the fused routine does not support
    // the size of the input panel, proceed with more generic code

    // static shared memory requirement, valid for:
    // dgeqr2_sm_kernel_batched, dgeqr2_column_sm_kernel_batched, dgeqr2_kernel_batched
    magma_int_t static_shmem = (BLOCK_SIZE + 1 ) * ( sizeof(double) + sizeof(double) );

    // dynamic shared memory
    magma_int_t dynamic_shmem_sm_kernel        = sizeof(double) * m * k;
    magma_int_t dynamic_shmem_column_sm_kernel = sizeof(double) * m;

    // total shared memory
    magma_int_t total_shmem_sm_kernel        = static_shmem + dynamic_shmem_sm_kernel;
    magma_int_t total_shmem_column_sm_kernel = static_shmem + dynamic_shmem_column_sm_kernel;

    // max. dynamic shared memory allowed per thread-block
    int shmem_max = 0;
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if ( total_shmem_sm_kernel <= shmem_max) {
        cudaFuncSetAttribute(dgeqr2_sm_kernel_batched, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_sm_kernel);
    }

    if ( total_shmem_column_sm_kernel <= shmem_max) {
        cudaFuncSetAttribute(dgeqr2_column_sm_kernel_batched, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shmem_column_sm_kernel);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000


    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(BLOCK_SIZE);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(1, 1, ibatch);

        if ( total_shmem_sm_kernel <= shmem_max ) {
            //load panel in shared memory and factorize it and copy back to gloabl memory
            //intend for small panel to avoid overfill of shared memory.
            //this kernel is composed of device routine and thus clean
            dgeqr2_sm_kernel_batched<<< grid, threads, sizeof(double)*(m*k), queue->cuda_stream() >>>
            (m, k, dA_array+i, Ai, Aj, ldda, dtau_array+i, taui);
        }
        else if ( total_shmem_column_sm_kernel <= shmem_max ) {
            //load one column vector in shared memory and householder it and used it to update trailing matrix which is global memory
            dgeqr2_column_sm_kernel_batched<<< grid, threads, sizeof(double)*(m), queue->cuda_stream() >>>
            (m, k, dA_array+i, Ai, Aj, ldda, dtau_array+i, taui);
        }
        else {
            //not use dynamic shared memory at all
            dgeqr2_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
            (m, k, dA_array+i, Ai, Aj, ldda, dtau_array+i, taui);
        }
    }

    return arginfo;
}
