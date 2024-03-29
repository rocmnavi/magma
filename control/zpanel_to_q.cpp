/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Mark Gates
       @precisions normal z -> s d c
*/
#include "magma_internal.h"

/***************************************************************************//**
    Put 0s in the upper triangular part of a panel and 1s on the diagonal.
    Stores previous values in work array, to be restored later with magma_zq_to_panel().
    
    @ingroup magma_panel2q
*******************************************************************************/
extern "C"
void magma_zpanel_to_q(magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work)
{
    magma_int_t i, j, k = 0;
    magmaDoubleComplex *col;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    
    if (uplo == MagmaUpper) {
        for (i = 0; i < ib; ++i) {
            col = A + i*lda;
            for (j = 0; j < i; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
            
            work[k] = col[i];
            col [j] = c_one;
            ++k;
        }
    }
    else {
        for (i=0; i < ib; ++i) {
            col = A + i*lda;
            work[k] = col[i];
            col [i] = c_one;
            ++k;
            for (j=i+1; j < ib; ++j) {
                work[k] = col[j];
                col [j] = c_zero;
                ++k;
            }
        }
    }
}


/***************************************************************************//**
    Restores a panel, after call to magma_zpanel_to_q().
    
    @ingroup magma_panel2q
*******************************************************************************/
extern "C"
void magma_zq_to_panel(magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work)
{
    magma_int_t i, j, k = 0;
    magmaDoubleComplex *col;
    
    if (uplo == MagmaUpper) {
        for (i = 0; i < ib; ++i) {
            col = A + i*lda;
            for (j = 0; j <= i; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
    else {
        for (i = 0; i < ib; ++i) {
            col = A + i*lda;
            for (j = i; j < ib; ++j) {
                col[j] = work[k];
                ++k;
            }
        }
    }
}
