/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETF2_NOPIV computes an LU factorization of a general m-by-n
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.
    
    This is a CPU-only (not accelerated) version.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    @ingroup magma_getf2_nopiv
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_nopiv(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *info)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;

    magma_int_t min_mn, m_j, n_j;
    magmaDoubleComplex inv_Ajj;
    magma_int_t i, j;
    double sfmin;

    A -= 1 + lda;

    /* Function Body */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Compute machine safe minimum */
    sfmin = lapackf77_dlamch("S");

    min_mn = min(m,n);
    for (j = 1; j <= min_mn; ++j) {
        /* Test for singularity. */
        if ( ! MAGMA_Z_EQUAL( *A(j,j), c_zero)) {
            /* Compute elements J+1:M of J-th column. */
            if (j < m) {
                if (MAGMA_Z_ABS( *A(j,j) ) >= sfmin) {
                    m_j = m - j;
                    inv_Ajj = MAGMA_Z_DIV(c_one, *A(j,j));
                    blasf77_zscal( &m_j, &inv_Ajj, A(j+1,j), &ione );
                }
                else {
                    m_j = m - j;
                    for (i = 1; i <= m_j; ++i) {
                        *A(j+i,j) = MAGMA_Z_DIV( *A(j+i,j), *A(j,j) );
                    }
                }
            }
        }
        else if (*info == 0) {
            *info = j;
        }

        if (j < min_mn) {
            /* Update trailing submatrix. */
            m_j = m - j;
            n_j = n - j;
            blasf77_zgeru( &m_j, &n_j, &c_neg_one,
                           A(j+1,j),   &ione,
                           A(j,j+1),   &lda,
                           A(j+1,j+1), &lda );
        }
    }

    return *info;
} /* magma_zgetf2_nopiv */
