/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from sparse/control/magma_zmcsrpass.cpp, normal z -> c, Fri Aug 25 13:19:12 2023
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Passes a CSR matrix to MAGMA.

    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows

    @param[in]
    n           magma_int_t
                number of columns

    @param[in]
    row         magma_index_t*
                row pointer

    @param[in]
    col         magma_index_t*
                column indices

    @param[in]
    val         magmaFloatComplex*
                array containing matrix entries

    @param[out]
    A           magma_c_matrix*
                matrix in magma sparse matrix format
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_ccsrset(
    magma_int_t m,
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_c_matrix *A,
    magma_queue_t queue )
{
    
    // make sure the target structure is empty
    magma_cmfree( A, queue );
    
    A->num_rows = m;
    A->num_cols = n;
    A->nnz = row[m];
    A->true_nnz = row[m];
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_CPU;
    A->val = val;
    A->col = col;
    A->row = row;
    A->fill_mode = MagmaFull;
    A->ownership = MagmaFalse;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA matrix to CSR structure.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                magma sparse matrix in CSR format

    @param[out]
    m           magma_int_t
                number of rows

    @param[out]
    n           magma_int_t
                number of columns

    @param[out]
    row         magma_index_t*
                row pointer

    @param[out]
    col         magma_index_t*
                column indices

    @param[out]
    val         magmaFloatComplex*
                array containing matrix entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_ccsrget(
    magma_c_matrix A,
    magma_int_t *m,
    magma_int_t *n,
    magma_index_t **row,
    magma_index_t **col,
    magmaFloatComplex **val,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix A_CPU={Magma_CSR}, A_CSR={Magma_CSR};
        
    if ( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ) {
        *m = A.num_rows;
        *n = A.num_cols;
        *val = A.val;
        *col = A.col;
        *row = A.row;
        A.ownership = MagmaFalse;
    } else {
        CHECK( magma_cmtransfer( A, &A_CPU, A.memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( A_CPU, &A_CSR, A_CPU.storage_type, Magma_CSR, queue ));
        CHECK( magma_ccsrget( A_CSR, m, n, row, col, val, queue ));
    }

cleanup:
    magma_cmfree( &A_CSR, queue );
    magma_cmfree( &A_CPU, queue );
    return info;
}
