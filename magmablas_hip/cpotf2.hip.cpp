#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zpotf2.hip.cpp, normal z -> c, Mon Oct 28 11:12:15 2024
*/
#include "magma_internal.h"

#define COMPLEX

#define cdotc_max_bs 512  // 512 is max threads for 1.x cards

void cpotf2_csscal( magma_int_t n, magmaFloatComplex *x, magma_int_t incx, magma_int_t* device_info, magma_queue_t queue );
void cpotf2_cdotc(magma_int_t n, magmaFloatComplex *x, magma_int_t incx, int step, magma_int_t* device_info, magma_queue_t queue );

#ifdef COMPLEX
void magmablas_clacgv( magma_int_t n, magmaFloatComplex *x, magma_int_t incx, magma_queue_t queue );
#endif


// TODO: this function could be in .cpp file -- it has no CUDA code in it.
/***************************************************************************//**
    Purpose
    -------

    cpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U**H * U,  if UPLO = MagmaUpper, or
        A = L  * L**H, if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.
    This version accepts a `device_info` argument for the status of the
    factorization. Errors in the arguments are captured in a return code.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0 and N <= 512.

    @param[in,out]
    dA      COMPLEX array, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U  or A = L * L**H.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[out]
    device_info  INTEGER (device memory)
      -     = 0: successful exit
      -     > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    @ingroup magma_potf2
*******************************************************************************/
extern "C" magma_int_t
magma_cpotf2_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t step, magma_int_t *device_info,
    magma_queue_t queue )
{
#define dA(i_, j_)  (dA + (i_) + (j_)*ldda)

#ifdef MAGMA_HAVE_CUDA
#define magma_cpotf2_gemv magmablas_cgemv

#else
#define magma_cpotf2_gemv magma_cgemv

#endif

    magma_int_t j;

    magma_int_t arginfo = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (n < 0 || n > cdotc_max_bs) {
        arginfo = -2;
    } else if (ldda < max(1,n)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }

    magmaFloatComplex alpha = MAGMA_C_NEG_ONE;
    magmaFloatComplex beta  = MAGMA_C_ONE;

    if (uplo == MagmaUpper) {
        for (j = 0; j < n; j++) {
            cpotf2_cdotc( j, dA(0,j), 1, step+j, device_info, queue ); // including cdotc product and update a(j,j)
            if (j < n) {
                #ifdef COMPLEX
                magmablas_clacgv( j, dA(0, j), 1, queue );
                #endif
                magma_cpotf2_gemv( MagmaTrans, j, n-j-1,
                             alpha, dA(0, j+1), ldda,
                                    dA(0, j),   1,
                             beta,  dA(j, j+1), ldda, queue );

                #ifdef COMPLEX
                magmablas_clacgv( j, dA(0, j), 1, queue );
                #endif
                cpotf2_csscal( n-j, dA(j,j), ldda, device_info, queue );
            }
        }
    }
    else {
        for (j = 0; j < n; j++) {
            cpotf2_cdotc( j, dA(j,0), ldda, step+j, device_info, queue ); // including cdotc product and update a(j,j)
            if (j < n) {
                #ifdef COMPLEX
                magmablas_clacgv( j, dA(j, 0), ldda, queue );
                #endif
                magma_cpotf2_gemv( MagmaNoTrans, n-j-1, j,
                             alpha, dA(j+1, 0), ldda,
                                    dA(j,0),    ldda,
                             beta,  dA(j+1, j), 1, queue );

                #ifdef COMPLEX
                magmablas_clacgv( j, dA(j, 0), ldda, queue );
                #endif
                cpotf2_csscal( n-j, dA(j,j), 1, device_info, queue );
            }
        }
    }

    return arginfo;

#undef magma_cpotf2_gemv
}

/***************************************************************************//**
    Purpose
    -------

    cpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U**H * U,  if UPLO = MagmaUpper, or
        A = L  * L**H, if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
      -     = MagmaUpper:  Upper triangular
      -     = MagmaLower:  Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0 and N <= 512.

    @param[in,out]
    dA      COMPLEX array, dimension (LDDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U  or A = L * L**H.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    @ingroup magma_potf2
*******************************************************************************/
extern "C" magma_int_t
magma_cpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0 || n > cdotc_max_bs) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0) {
        return *info;
    }

    magma_int_t* device_info;
    magma_imalloc(&device_info, 1);
    magma_memset_async((void*)device_info, 0, sizeof(magma_int_t), queue);

    magma_cpotf2_native(uplo, n, dA, ldda, 0, device_info, queue );

    magma_getvector(1, sizeof(magma_int_t), device_info, 1, info, 1, queue);
    magma_free(device_info);

    return *info;

#undef magma_cpotf2_gemv
}

#define csscal_bs  32
#define cdotc_bs  512
#define clacgv_bs 512


__global__ void cpotf2_cdotc_kernel(int n, magmaFloatComplex *x, int incx, int threadSize, int step, magma_int_t* device_info)
{

    // dynamically allocated shared memory, set to size number of threads when the kernel is launched.
    // See CUDA Guide B.2.3
    HIP_DYNAMIC_SHARED( float, shared_data)

    // check for info from a previous factorization
    if( step > 0 && device_info[0] != 0 ) return;

    int tx = threadIdx.x;
    float *sdata = shared_data;

    magmaFloatComplex res = MAGMA_C_ZERO;

    if (tx < n) {
        res = x[tx*incx];
    }

    sdata[tx] = MAGMA_C_REAL(res * MAGMA_C_CONJ(res));

    __syncthreads();

    for (int s = blockDim.x/2; s > 32; s >>= 1 ) {
        if (tx < s) {
            sdata[tx] += sdata[tx+s];
        }
        __syncthreads();
    }

    if (tx < 32) {
        volatile float* smem = sdata;
        smem[tx] += smem[tx+32];
        smem[tx] += smem[tx+16];
        smem[tx] += smem[tx+8];
        smem[tx] += smem[tx+4];
        smem[tx] += smem[tx+2];
        smem[tx] += smem[tx+1];
    }

    if (tx == 0) {
        float xreal = MAGMA_C_REAL(x[n*incx]) - sdata[0];
        if(xreal < 0) {
            device_info[0] = (magma_int_t)step;
        }
        else{
            x[n*incx] = MAGMA_C_MAKE( sqrt(xreal), 0 );
        }
    }
}

void cpotf2_cdotc(
    magma_int_t n, magmaFloatComplex *x, magma_int_t incx,
    int step, magma_int_t* device_info, magma_queue_t queue )
{
    /*
    Specialized Cdotc
    1) performs cdotc sum = x[0:n-1]*conj(x[0:n-1])
    2) updates x[n] = sqrt(x[n]-sum);

    */
    if (n > cdotc_max_bs) {
        fprintf( stderr, "n = %lld > %lld is not supported in cpotf2_cdotc\n",
                 (long long) n, (long long) cdotc_max_bs );
        return;
    }
    int threadSize;

    if (n <= 1024 && n > 512) {
        threadSize = 1024;
    }
    else if (n <= 512 && n > 256 ) {
        threadSize = 512;
    }
    else if (n <= 256 && n > 128) {
        threadSize = 256;
    }
    else if (n <= 128 && n > 64) {
        threadSize = 128;
    }
    else {
        threadSize = 64;
    }

    size_t shmem = threadSize * sizeof(float);
    hipLaunchKernelGGL(cpotf2_cdotc_kernel, dim3(1), dim3(threadSize), shmem, queue->hip_stream() , n, x, incx, threadSize, step, device_info);
}

__global__ void cpotf2_csscal_kernel(int n, magmaFloatComplex *x, int incx, magma_int_t* device_info)
{
    int id = blockIdx.x * csscal_bs + threadIdx.x;

    // check for info
    if(device_info[0] != 0) return;

    __shared__ magmaFloatComplex factor;

    if (threadIdx.x == 0) {
        factor = MAGMA_C_MAKE(1.0/MAGMA_C_REAL(x[0]), 0.0);
    }

    __syncthreads();

    if ( id < n && id > 0) {
        x[id*incx] = x[id*incx] * factor;
    }
}


void cpotf2_csscal(
    magma_int_t n, magmaFloatComplex *x, magma_int_t incx,
    magma_int_t* device_info, magma_queue_t queue )
{
    /* Specialized csscal perform x[1:n-1] / x[0] */
    dim3 threads(csscal_bs, 1, 1);
    int num_blocks = magma_ceildiv( n, csscal_bs );
    dim3 grid(num_blocks,1);
    hipLaunchKernelGGL(cpotf2_csscal_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() , n, x, incx, device_info);
}


#ifdef COMPLEX

__global__ void cpotf2_clacgv_kernel(int n, magmaFloatComplex *x, int incx)
{
    int id = blockIdx.x * clacgv_bs + threadIdx.x;

    if ( id < n ) {
        x[id*incx] = MAGMA_C_CONJ(x[id*incx]);
    }
}


/***************************************************************************//**
    Purpose
    -------

    CLACGV conjugates a complex vector of length N.

    Arguments
    ---------

    @param[in]
    n       INTEGER
            The length of the vector X.  N >= 0.

    @param[in,out]
    x       COMPLEX array, dimension (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    @param[in]
    incx    INTEGER
            The spacing between successive elements of X.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lacgv
*******************************************************************************/
void magmablas_clacgv(
    magma_int_t n, magmaFloatComplex *x, magma_int_t incx,
    magma_queue_t queue )
{
    if(n <= 0) return;

    dim3 threads(clacgv_bs, 1, 1);
    int num_blocks = magma_ceildiv( n, clacgv_bs );
    dim3 grid(num_blocks,1);
    hipLaunchKernelGGL(cpotf2_clacgv_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() , n, x, incx);
}

#endif // COMPLEX