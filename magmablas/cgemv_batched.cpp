/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas/zgemv_batched.cpp, normal z -> c, Fri Aug 25 13:18:47 2023

       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    CGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    This is the batch version of the routine, using pointer-to-pointer (P2P)
    interface. All matrices and vectors must have the same dimension(s).

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A

    @param[in]
    alpha   COMPLEX
            On entry, ALPHA specifies the scalar alpha.


    @param[in]
    dA_array     Array of pointers, dimension (batchCount).
             Each is a COMPLEX array A of DIMENSION ( ldda, n ) on the GPU

    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    beta    COMPLEX
            On entry, ALPHA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy_array     Array of pointers, dimension (batchCount).
            Each is a COMPLEX array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv_batched
*******************************************************************************/
extern "C" void
magmablas_cgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dx_array, magma_int_t incx,
    const magmaFloatComplex beta,
    magmaFloatComplex** dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if(m == n && n <= 32) {
        info = magmablas_cgemv_batched_smallsq(
                trans, n,
                alpha, dA_array, ldda,
                       dx_array, incx,
                beta,  dy_array, incy,
                batchCount, queue);
        if(info == 0) return;
    }

    magmablas_cgemv_batched_core(
        trans, m, n,
        alpha, dA_array, NULL, ldda, 0,
               dx_array, NULL, incx, 0,
        beta,  dy_array, NULL, incy, 0,
        batchCount, queue);
}


/***************************************************************************//**
    Purpose
    -------
    CGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    This is the batch version of the routine, using "pointer + stride"
    interface. All matrices and vectors must have the same dimension(s).

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A

    @param[in]
    alpha   COMPLEX
            On entry, ALPHA specifies the scalar alpha.


    @param[in]
    dA      Pointer to the first COMPLEX array 'A' of DIMENSION ( ldda, n )
            in the batch

    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    strideA INTEGER
            specifies the distance between two consecutive matrices in the batch.

    @param[in]
    dx      Pointer to the first COMPLEX array 'x' in the batch, of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    stridex INTEGER
            Specifies the distance between two consecutive vectors in the batch

    @param[in]
    beta    COMPLEX
            On entry, ALPHA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      Pointer to the first COMPLEX array 'y' in the batch, of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    stridey INTEGER
            Specifies the distance between two consecutive vectors in the batch

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv_batched
*******************************************************************************/
extern "C" void
magmablas_cgemv_batched_strided(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    const magmaFloatComplex alpha,
    const magmaFloatComplex* dA, magma_int_t ldda, magma_int_t strideA,
    const magmaFloatComplex* dx, magma_int_t incx, magma_int_t stridex,
    const magmaFloatComplex beta,
    magmaFloatComplex* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if(m == n && n <= 32) {
        info = magmablas_cgemv_batched_strided_smallsq(
                    trans, n,
                    alpha, dA, ldda, strideA,
                           dx, incx, stridex,
                    beta,  dy, incy, stridey,
                    batchCount, queue);
        if( info == 0 ) return;
    }
    magmablas_cgemv_batched_core(
        trans, m, n,
        alpha, NULL, dA, ldda, strideA,
               NULL, dx, incx, stridex,
        beta,  NULL, dy, incy, stridey,
        batchCount, queue);
}