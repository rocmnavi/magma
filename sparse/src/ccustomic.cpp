/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Hartwig Anzt

       @generated from sparse/src/zcustomic.cpp, normal z -> c, Fri Aug 25 13:19:27 2023
*/
#include "magmasparse_internal.h"


#include "../blas/magma_trisolve.h"

#define COMPLEX

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#ifdef MAGMA_HAVE_HIP
  #define hipblasComplex hipFloatComplex
#endif

/**
    Purpose
    -------

    Reads in an Incomplete Cholesky preconditioner.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_ccustomicsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix hA={Magma_CSR};
    char preconditionermatrix[255];
    
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ICT.mtx" );
    
    CHECK( magma_c_csr_mtx( &hA, preconditionermatrix , queue) );
    
    
    // for CUSPARSE
    CHECK( magma_cmtransfer( hA, &precond->M, Magma_CPU, Magma_DEV , queue ));

        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_cmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_cmtranspose( precond->L, &(precond->U), queue ));

    // extract the diagonal of L into precond->d
    CHECK( magma_cjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_cvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));

    // extract the diagonal of U into precond->d2
    CHECK( magma_cjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_cvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));

    CHECK(magma_ctrisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_ctrisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
        
cleanup:
    magma_cmfree( &hA, queue );
    
    return info;
}
    
