/*
   -- MAGMA (version 2.7.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date August 2023

   @generated from magmablas/zposv_irgmres_kernels.cu, normal z -> c, Fri Aug 25 13:18:58 2023
   @author Ahmad Abdelfattah
 */

#include "magma_internal.h"
#include "batched_kernel_param.h"

////////////////////////////////////////////////////////////////////////////////
__global__ void
extract_diag_sqrt_kernel(int min_mn, magmaFloatComplex* dA, int ldda, float* dD, int incd)
{
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;
    if( gtx < min_mn ) {
        dD[gtx * incd] = sqrt( MAGMA_C_REAL( dA[gtx * ldda + gtx] ) );
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int DIMX, int DIMY>
__global__ void
cscal_shift_hpd_kernel(
        magma_uplo_t uplo, int n,
        magmaFloatComplex* dA, int ldda,
        float* dD, int incd,
        float miu, float cn, float eps)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int gbx = blockIdx.x * blockDim.x;
    const int gby = blockIdx.y * blockDim.y;

    const int gtx = gbx + tx;
    const int gty = gby + ty;

    __shared__ magmaFloatComplex sD_row[DIMX];
    __shared__ magmaFloatComplex sD_col[DIMY];

    magmaFloatComplex rA = MAGMA_C_ZERO;
    float rTmp = MAGMA_D_ZERO;
    // read the corresponding segments from diagonal vector
    // for pre-multiplication
    if(ty == 0 && gtx < n) {
        rTmp = dD[gtx * incd];
        sD_row[ tx ] = MAGMA_C_DIV( MAGMA_C_ONE, MAGMA_C_MAKE(rTmp, 1.) );
    }

    // for post multiplication
    const int y_length = min(DIMY, n - gby);
    if( ty == 1 && tx < y_length ) {
        rTmp = dD[ (gby+tx) * incd];
        sD_col[ tx ] = MAGMA_C_DIV( MAGMA_C_ONE, MAGMA_C_MAKE(rTmp, 1) );
    }
    __syncthreads();

    // read
    if(gtx < n && gty < n) {
        rA = dA[gty * ldda + gtx];
    }

    // D^-1 * A -- multiply scale rows
    rA *= sD_row[ tx ];

    // rA * D^-1 -- scale columns
    rA *= sD_col[ ty ];

    rA  = (gtx == gty) ? MAGMA_C_MAKE(1 + (cn*eps), 0. )  : rA;
    rA *= MAGMA_C_MAKE( miu, 0. );

    // write
    if(gtx < n && gty < n) {
        dA[gty * ldda + gtx] = rA;
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void
dimv_kernel(
        int n,
        magmaFloatComplex alpha, magmaFloatComplex *dD, int incd,
                                  magmaFloatComplex *dx, int incx,
        magmaFloatComplex beta,  magmaFloatComplex *dy, int incy,
        bool invert_diagonal)
{
    const int gtx = blockIdx.x * blockDim.x + threadIdx.x;

    magmaFloatComplex rA = MAGMA_C_ZERO;
    if(beta != MAGMA_C_ZERO) {
        if(gtx < n)
            rA = beta * dy[ gtx * incy];
    }

    magmaFloatComplex rTmp = MAGMA_C_ZERO;
    if( gtx < n) {
        rTmp += (invert_diagonal) ? MAGMA_C_DIV(MAGMA_C_ONE, dD[gtx * incd]) * dx[gtx * incx] :
                                    dD[gtx*incd] * dx[gtx * incx];
        rTmp *= alpha;
        dy[gtx * incy] = rA + rTmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
// extract the diagonal of an mxn matrix, and write its sqrt to a vector
extern "C"
void
magmablas_cextract_diag_sqrt(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex* dA, magma_int_t ldda,
    float* dD, magma_int_t incd,
    magma_queue_t queue)
{
    const int bx = 256;
    const int min_mn = min(m, n);
    const int nblocks = magma_ceildiv(min_mn, 256);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(bx, 1, 1);
    extract_diag_sqrt_kernel<<<grid, threads, 0, queue->cuda_stream()>>>
    (min_mn, dA, ldda, dD, incd);
}

////////////////////////////////////////////////////////////////////////////////
// two-sided diagonal scaling and shifting for hpd matrices
//  ** A becomes D^-1 * A * D^-1, where D diag( sqrt(a(i,i)) )
//  ** Diagonal elements are also shifted by cn * eps, where cn is a constant
//     of choice and eps is the machine epsilon
//  ** An optional additional scaling by miu is also available
//
// Please see for more details:
//  ** "Exploiting Lower Precision Arithmetic in Solving Symmetric Positive
//      Definite Linear Systems and Least Squares Problems", by Higham et al.
//      http://eprints.maths.manchester.ac.uk/2771/
//
// this kernel ignores uplo and scales the whole matrix
// TODO: scale the upper or the lower triangular part only
extern "C"
void
magmablas_cscal_shift_hpd(
    magma_uplo_t uplo, int n,
    magmaFloatComplex* dA, int ldda,
    float* dD, int incd,
    float miu, float cn, float eps,
    magma_queue_t queue)
{
    const int DIMX = 32;
    const int DIMY = 8;

    // required check for the kernel to work properly
    assert(DIMX >= DIMY);

    dim3 threads(DIMX, DIMY, 1);
    dim3 grid(magma_ceildiv(n, DIMX), magma_ceildiv(n, DIMY), 1);

    cscal_shift_hpd_kernel<DIMX, DIMY><<<grid, threads, 0, queue->cuda_stream()>>>
    (uplo, n, dA, ldda, dD, incd, miu, cn, eps);
}

////////////////////////////////////////////////////////////////////////////////
// Computes inverse(diagonal-matrix) x vector
// Diagonal matrix is stored as a dense vector
// operation can be done in-place
extern "C"
void
magmablas_cdimv_invert(
        magma_int_t n,
        magmaFloatComplex alpha, magmaFloatComplex* dD, magma_int_t incd,
                                  magmaFloatComplex* dx, magma_int_t incx,
        magmaFloatComplex beta,  magmaFloatComplex* dy, magma_int_t incy,
        magma_queue_t queue)
{
    const int nthreads = 256;
    dim3 threads(nthreads, 1, 1);
    dim3 grid( magma_ceildiv(n, nthreads), 1, 1);

    dimv_kernel<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, alpha, dD, incd, dx, incx, beta,  dy, incy, true);
}