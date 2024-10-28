#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Stan Tomov

       @generated from magmablas_hip/zgbsv_kernels.hip.cpp, normal z -> d, Mon Oct 28 11:12:15 2024
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"
#include "dgbtf2_devicefunc.hip.hpp"

#define PRECISION_d

#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)        (max(1,64/(M)))
#else
#define NTCOL(M)        (max(1,64/(M)))
#endif

#define SLDA(N)    ((N)+1)

////////////////////////////////////////////////////////////////////////////////
template<int MAX_THREADS>
__global__ __launch_bounds__(MAX_THREADS * NTCOL(MAX_THREADS))
void
dgbsv_batched_kernel_fused_sm(
    int n, int kl, int ku, int nrhs,
    double** dA_array, int ldda, magma_int_t** ipiv_array,
    double** dB_array, int lddb,
    magma_int_t *info_array, int batchCount)
{
#define sA(i,j)        sA[(j)*slda + (i)]
#define sB(i,j)        sB[(j)*sldb + (i)]
#define dA(i,j)        dA[(j)*ldda + (i)]
#define dB(i,j)        dB[(j)*lddb + (i)]

    HIP_DYNAMIC_SHARED( double, zdata)
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x; // not necessarily equal to MAX_THREADS
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount) return;

    const int kv     = kl + ku;
    const int nband  = (kl + 1 + kv);
    const int slda   = SLDA(nband);
    const int sldb   = SLDA(n);
    const int slda_1 = slda-1;

    double* dA = dA_array[batchid];
    double* dB = dB_array[batchid];
    int linfo = 0;

    // shared memory pointers
    double *sA = (double*)(zdata);
    double *sB = sA + blockDim.y * n * slda;
    double* dsx            = (double*)(sB + blockDim.y * nrhs * sldb);
    int* sipiv             = (int*)(dsx + blockDim.y * (kl+1));
    sA    += ty * n * slda;
    sB    += ty * nrhs * sldb;
    dsx   += ty * (kl+1);
    sipiv += ty * n;

    // init sA
    for(int i = tx; i < n*slda; i+=ntx) {
        sA[i] = MAGMA_D_ZERO;
    }
    __syncthreads();

    // read A & B
    read_sAB(nband, n, kl, ku, dA, ldda, sA, slda, ntx, tx);
    read_sB(n, nrhs, dB, lddb, sB, sldb, ntx, tx );
    __syncthreads();


    // factorize + forward solve
    int ju = 0;
    for(int j = 0; j < n; j++) {
        // idamax
        int kn = 1 + min( kl, n-j-1 ); // diagonal and subdiagonal(s)
        if(tx < kn) {
            dsx[ tx ] = fabs(MAGMA_D_REAL( sA(kv+tx,j) )) + fabs(MAGMA_D_IMAG( sA(kv+tx,j) ));
        }
        __syncthreads();

        double rx_abs_max = dsx[0];
        int    jp       = 0;
        for(int i = 1; i < kn; i++) {
            if( dsx[i] > rx_abs_max ) {
                rx_abs_max = dsx[i];
                jp         = i;
            }
        }

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (j+1) : linfo;

        if(tx == 0) {
            sipiv[j] = jp + j + 1;    // +1 for fortran indexing
        }

        ju = max(ju, min(j+ku+jp, n-1));
        int swap_len = ju - j + 1;

        // swap
        if( !(jp == 0) ) {
            // swap A
            double tmp;
            double *sR1 = &sA(kv   ,j);
            double *sR2 = &sA(kv+jp,j);
            for(int i = tx; i < swap_len; i+=ntx) {
                tmp             = sR1[i * slda_1];
                sR1[i * slda_1] = sR2[i * slda_1];
                sR2[i * slda_1] = tmp;
            }

            // swap B
            for(int i = tx; i < nrhs; i+=ntx) {
                tmp         = sB(   j, i);
                sB(   j, i) = sB(jp+j, i);
                sB(jp+j, i) = tmp;
            }
        }
        __syncthreads();

        // scal
        double reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_D_ONE : MAGMA_D_DIV(MAGMA_D_ONE, sA(kv,j) );
        for(int i = tx; i < (kn-1); i+=ntx) {
            sA(kv+1+i, j) *= reg;
        }
        __syncthreads();

        // ger
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_D_ZERO : MAGMA_D_ONE;
        double *sU  = &sA(kv,j);
        double *sV  = &sA(kv+1,j);

        if( tx < (kn-1) ) {
            for(int jj = 1; jj < swap_len; jj++) {
                sV[jj * slda_1 + tx] -= sV[tx] * sU[jj * slda_1 + 0] * reg;
            }

            // apply the current column to B
            for(int jj = 0; jj < nrhs; jj++) {
                sB(j + tx + 1, jj) -= sV[tx] * sB(j,jj) * reg;
            }
        }
        __syncthreads();
    }

    // backward solv
    for(int j = n-1; j >= 0; j--) {
        int nupdates = min(kv, j);
        for(int rhs = 0; rhs < nrhs; rhs++) {
            double s = sB(j,rhs) * MAGMA_D_DIV(MAGMA_D_ONE, sA(kv,j));
            __syncthreads();

            if(tx == 0) sB(j,rhs) = s;
            for(int i = tx; i < nupdates ; i+= ntx) {
                sB(j-i-1,rhs) -= s * sA(kv-i-1,j);
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // write info
    if(tx == 0) info_array[batchid] = linfo;

    // write pivot
    magma_int_t* ipiv = ipiv_array[batchid];
    for(int i = tx; i < n; i+=ntx) {
        ipiv[i] = (magma_int_t)sipiv[i];
    }

    write_sAB(nband, n, kl, ku, sA, slda, dA, ldda, ntx, tx);
    write_sB(n, nrhs, sB, sldb, dB, lddb, ntx, tx );

#undef sA
#undef sB
#undef dA
#undef dB
}

////////////////////////////////////////////////////////////////////////////////
template<int MAX_THREADS>
magma_int_t
magma_dgbsv_batched_fused_sm_kernel_driver(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    double** dA_array, magma_int_t ldda, magma_int_t** ipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    magma_int_t kv      = kl + ku;
    magma_int_t nband   = kv + 1 + kl;
    magma_int_t slda   = SLDA(nband);
    magma_int_t sldb   = SLDA(n);

    nthreads = max( nthreads, (kl + 1) );
    ntcol    = max(1, ntcol);

    magma_int_t shmem  = 0;
    shmem += slda * n     * sizeof(double); // sA
    shmem += sldb * nrhs  * sizeof(double); // rhs (sB)
    shmem += (kl + 1)     * sizeof(double);             // dsx
    shmem += n            * sizeof(int);        // pivot
    shmem *= ntcol;

    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 threads(nthreads, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    hipDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(dgbsv_batched_kernel_fused_sm<MAX_THREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&n, &kl, &ku, &nrhs, &dA_array, &ldda, &ipiv_array, &dB_array, &lddb, &info_array, &batchCount};
    hipError_t e = hipLaunchKernel((void*)dgbsv_batched_kernel_fused_sm<MAX_THREADS>, grid, threads, kernel_args, shmem, queue->hip_stream());
    if( e != hipSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, hipGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    DGBSV computes the solution to a system of linear equations
    A * X = B, where A is a band matrix of order N with KL subdiagonals
    and KU superdiagonals, and X and B are N-by-NRHS matrices.

    The LU decomposition with partial pivoting and row interchanges is
    used to factor A as A = L * U, where L is a product of permutation
    and unit lower triangular matrices with KL subdiagonals, and U is
    upper triangular with KL+KU superdiagonals.  The factored form of A
    is then used to solve the system of equations A * X = B.

    This is the batched version of the routine.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    kl      INTEGER
            The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    ku      INTEGER
            The number of superdiagonals within the band of A.  KL >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA_array    Array of pointers, dimension (batchCount).
                Each contains the details of the LU factorization of the band matrix A,
                as computed by DGBTRF.  U is stored as an upper triangular band
                matrix with KL+KU superdiagonals in rows 1 to KL+KU+1, and
                the multipliers used during the factorization are stored in
                rows KL+KU+2 to 2*KL+KU+1.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= (2*KL+KU+1).

    @param[in]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[in,out]
    dB_array    Array of pointers, dimension (batchCount).
                Each is a DOUBLE PRECISION array, dimension (LDB,NRHS)
                On entry, the right hand side matrix B.
                On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of each array B.  LDDB >= max(1, N).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    nthreads    INTEGER
                The number of threads assigned to a single matrix.
                nthreads >= (KL+1)

    @param[in]
    ntcol       INTEGER
                The number of concurrent factorizations in a thread-block
                ntcol >= 1

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_dgbsv_batched_fused_sm(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    double** dA_array, magma_int_t ldda, magma_int_t** ipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    magma_int_t kv      = kl + ku;

    if ( n < 0 )
        arginfo = -1;
    else if ( kl < 0 )
        arginfo = -2;
    else if ( ku < 0 )
        arginfo = -3;
    else if (nrhs < 0)
        arginfo = -4;
    else if ( ldda < (kl+kv+1) )
        arginfo = -6;
    else if ( lddb < n)
        arginfo = -9;
    else if ( batchCount < 0 )
        arginfo = -13;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( n == 0 || nrhs == 0 || batchCount == 0 ) return 0;

    magma_int_t nthread32 = magma_roundup(nthreads, 32);

    switch(nthread32) {
        case   32: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver<  32>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   64: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver<  64>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   96: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver<  96>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  128: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 128>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  160: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 160>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  192: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 192>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  224: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 224>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  256: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 256>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  288: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 288>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  320: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 320>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  352: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 352>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  384: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 384>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  416: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 416>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  448: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 448>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  480: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 480>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  512: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 512>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  544: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 544>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  576: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 576>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  608: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 608>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  640: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 640>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  672: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 672>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  704: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 704>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  736: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 736>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  768: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 768>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  800: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 800>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  832: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 832>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  864: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 864>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  896: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 896>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  928: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 928>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  960: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 960>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  992: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver< 992>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        case 1024: arginfo = magma_dgbsv_batched_fused_sm_kernel_driver<1024>(n, kl, ku, nrhs, dA_array, ldda, ipiv_array, dB_array, lddb, info_array, nthreads, ntcol, batchCount, queue ); break;
        default: arginfo = -100;
    }

    return arginfo;
}
