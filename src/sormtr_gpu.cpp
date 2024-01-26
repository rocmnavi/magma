/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Stan Tomov

       @generated from src/zunmtr_gpu.cpp, normal z -> s, Thu Jan 25 23:02:50 2024

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    SORMTR overwrites the general real M-by-N matrix C with

                                SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:       Q * C               C * Q
    TRANS = MagmaTrans:    Q**H * C            C * Q**H

    where Q is a real orthogonal matrix of order nq,
    with nq = m if SIDE = MagmaLeft
    and  nq = n if SIDE = MagmaRight. Q is defined as the product of
    nq-1 elementary reflectors, as returned by SSYTRD:

    if UPLO = MagmaUpper, Q = H(nq-1) . . . H(2) H(1);

    if UPLO = MagmaLower, Q = H(1) H(2) . . . H(nq-1).

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply Q or Q**H from the Left;
      -     = MagmaRight:     apply Q or Q**H from the Right.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper: Upper triangle of A contains elementary reflectors
                   from SSYTRD;
      -     = MagmaLower: Lower triangle of A contains elementary reflectors
                   from SSYTRD.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = MagmaTrans: Conjugate transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in,out]
    dA      REAL array, dimension
                                 (LDDA,M) if SIDE = MagmaLeft
                                 (LDDA,N) if SIDE = MagmaRight
            The vectors which define the elementary reflectors, as
            returned by SSYTRD_GPU. On output the diagonal, the subdiagonal
            and the upper part (UPLO=MagmaLower) or lower part (UPLO=MagmaUpper)
            are destroyed.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.
            If SIDE = MagmaLeft,  LDDA >= max(1,M);
            if SIDE = MagmaRight, LDDA >= max(1,N).

    @param[in]
    tau     REAL array, dimension
                                 (M-1) if SIDE = MagmaLeft
                                 (N-1) if SIDE = MagmaRight
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SSYTRD.

    @param[in,out]
    dC      REAL array, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    @param[in]
    lddc    INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    @param[in]
    wA      REAL array, dimension
                                 (LDWA,M) if SIDE = MagmaLeft
                                 (LDWA,N) if SIDE = MagmaRight
            The vectors which define the elementary reflectors, as
            returned by SSYTRD_GPU.
            (A copy of the upper or lower part of dA, on the host.)

    @param[in]
    ldwa    INTEGER
            The leading dimension of the array wA.
            If SIDE = MagmaLeft,  LDWA >= max(1,M);
            if SIDE = MagmaRight, LDWA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_unmtr
*******************************************************************************/
extern "C" magma_int_t
magma_sormtr_gpu(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA,    magma_int_t ldda,
    float   *tau,
    magmaFloat_ptr dC,    magma_int_t lddc,
    const float *wA, magma_int_t ldwa,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dC(i_,j_) (dC + (i_) + (j_)*lddc)
    #define wA(i_,j_) (wA + (i_) + (j_)*ldwa)
    
    magma_int_t i1, i2, mi, ni, nq;
    magma_int_t iinfo;

    *info = 0;
    bool left   = (side == MagmaLeft);
    bool upper  = (uplo == MagmaUpper);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        //nw = n;
    } else {
        nq = n;
        //nw = m;
    }
    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! upper && uplo != MagmaLower) {
        *info = -2;
    } else if (trans != MagmaNoTrans &&
               trans != MagmaTrans) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nq == 1) {
        return *info;
    }

    if (left) {
        mi = m - 1;
        ni = n;
    } else {
        mi = m;
        ni = n - 1;
    }

    if (upper) {
        magma_sormql2_gpu(side, trans, mi, ni, nq-1, dA(0,1), ldda, tau,
                          dC, lddc, wA(0,1), ldwa, &iinfo);
    }
    else {
        /* Q was determined by a call to SSYTRD with UPLO = MagmaLower */
        if (left) {
            i1 = 1;
            i2 = 0;
        } else {
            i1 = 0;
            i2 = 1;
        }
        magma_sormqr2_gpu(side, trans, mi, ni, nq-1, dA(1,0), ldda, tau,
                          dC(i1,i2), lddc, wA(1,0), ldwa, &iinfo);
    }

    return *info;
} /* magma_sormtr */
