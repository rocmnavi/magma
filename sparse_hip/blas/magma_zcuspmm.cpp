/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

#if CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
#define hipsparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA, drowA, dcolA,        \
                            descrB, nnzB, drowB, dcolB, descrC, drowC, nnzTotal )               \
    {                                                                                           \
        hipsparseMatDescr_t descrD;                                                              \
        int nnzD, *drowD, *dcolD;                                                               \
        csrgemm2Info_t linfo = NULL;                                                            \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipblasDoubleComplex alpha = MAGMA_Z_ONE, *beta = NULL;                                      \
        hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST);                             \
        hipsparseCreateCsrgemm2Info(&linfo);                                                     \
        hipsparseZcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha,                                \
                                        descrA, nnzA, drowA, dcolA,                             \
                                        descrB, nnzB, drowB, dcolB,                             \
                                        beta,                                                   \
                                        descrD, nnzD, drowD, dcolD,                             \
                                        linfo, &bufsize);                                       \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseXcsrgemm2Nnz(handle, m, n, k,                                                   \
                             descrA, nnzA, drowA, dcolA,                                        \
                             descrB, nnzB, drowB, dcolB,                                        \
                             descrD, nnzD, drowD, dcolD,                                        \
                             descrC, drowC, nnzTotal,                                           \
                             linfo, buf);                                                       \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

// todo: info and buf are passed from the above function
// also at the end destroy info: hipsparseDestroyCsrgemm2Info(info);
#define hipsparseZcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, dvalA, drowA, dcolA,    \
                         descrB, nnzB, dvalB, drowB, dcolB, descrC, dvalC, drowC, dcolC )       \
    {                                                                                           \
        hipsparseMatDescr_t descrD;                                                              \
        int nnzD, *drowD, *dcolD;                                                               \
        hipblasDoubleComplex *dvalD, alpha = MAGMA_Z_ONE, *beta = NULL;                              \
        void *buf;                                                                              \
        csrgemm2Info_t linfo = NULL;                                                            \
        printf("hipsparseZcsrgemm bufsize = %d\n", -1);                                          \
        hipsparseZcsrgemm2(handle, m, n, k, &alpha,                                              \
                          descrA, nnzA, dvalA, drowA, dcolA,                                    \
                          descrB, nnzB, dvalB, drowB, dcolB,                                    \
                          beta,                                                                 \
                          descrD, nnzD, dvalD, drowD, dcolD,                                    \
                          descrC, dvalC, drowC, dcolC,                                          \
                          linfo, buf);                                                          \
    }
#endif


/**
    Purpose
    -------

    This is an interface to the cuSPARSE routine csrmm computing the product
    of two sparse matrices stored in csr format.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix

    @param[in]
    B           magma_z_matrix
                input matrix

    @param[out]
    AB          magma_z_matrix*
                output matrix AB = A * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcuspmm(
    magma_z_matrix A, magma_z_matrix B,
    magma_z_matrix *AB,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    
    magma_z_matrix C={Magma_CSR};
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.storage_type = A.storage_type;
    C.memory_location = A.memory_location;
    C.fill_mode = MagmaFull;
    
    C.val = NULL;
    C.col = NULL;
    C.row = NULL;
    C.rowidx = NULL;
    C.blockinfo = NULL;
    C.diag = NULL;
    C.dval = NULL;
    C.dcol = NULL;
    C.drow = NULL;
    C.drowidx = NULL;
    C.ddiag = NULL;

#if CUDA_VERSION < 12000 || defined(MAGMA_HAVE_HIP)
    
    magma_index_t base_t, nnz_t, baseC;
    
    hipsparseHandle_t handle=NULL;
    hipsparseMatDescr_t descrA=NULL;
    hipsparseMatDescr_t descrB=NULL;
    hipsparseMatDescr_t descrC=NULL;
    
    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) )
    {
        // CUSPARSE context /
        CHECK_CUSPARSE( hipsparseCreate( &handle ));
        CHECK_CUSPARSE( hipsparseSetStream( handle, queue->hip_stream() ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrA ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrB ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrC ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrB, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrC, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, HIPSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrB, HIPSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrC, HIPSPARSE_INDEX_BASE_ZERO ));

        // nnzTotalDevHostPtr points to host memory
        magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &C.nnz;
        CHECK_CUSPARSE( hipsparseSetPointerMode( handle, HIPSPARSE_POINTER_MODE_HOST ));
        CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));
        hipsparseXcsrgemmNnz( handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             A.num_rows, B.num_cols, A.num_cols,
                             descrA, A.nnz, A.drow, A.dcol,
                             descrB, B.nnz, B.drow, B.dcol,
                             descrC, C.drow, nnzTotalDevHostPtr );
        if (NULL != nnzTotalDevHostPtr) {
            C.nnz = *nnzTotalDevHostPtr;
        } else {
            // workaround as nnz and base C are magma_int_t
            magma_index_getvector( 1, C.drow+C.num_rows, 1, &nnz_t, 1, queue );
            magma_index_getvector( 1, C.drow,   1, &base_t,    1, queue );
            C.nnz = (magma_int_t) nnz_t;
            baseC = (magma_int_t) base_t;
            C.nnz -= baseC;
        }
        CHECK( magma_index_malloc( &C.dcol, C.nnz ));
        CHECK( magma_zmalloc( &C.dval, C.nnz ));
        #ifdef MAGMA_HAVE_HIP
        hipsparseZcsrgemm( handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          A.num_rows, B.num_cols, A.num_cols,
                          descrA, A.nnz,
                          (const hipDoubleComplex*)A.dval, A.drow, A.dcol,
                          descrB, B.nnz,
                          (const hipDoubleComplex*)B.dval, B.drow, B.dcol,
                          descrC,
                          (hipDoubleComplex*)C.dval, C.drow, C.dcol );

        #else
        hipsparseZcsrgemm( handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          A.num_rows, B.num_cols, A.num_cols,
                          descrA, A.nnz,
                          A.dval, A.drow, A.dcol,
                          descrB, B.nnz,
                          B.dval, B.drow, B.dcol,
                          descrC,
                          C.dval, C.drow, C.dcol );
        #endif
        // end CUSPARSE context //
        magma_queue_sync( queue );
        CHECK( magma_zmtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
cleanup:
    hipsparseDestroyMatDescr( descrA );
    hipsparseDestroyMatDescr( descrB );
    hipsparseDestroyMatDescr( descrC );
    hipsparseDestroy( handle );
    magma_zmfree( &C, queue );
    return info;

#else
    // ================================================================================
    magma_index_t base_t, nnz_t, baseC;

    hipsparseHandle_t handle=NULL;
    hipsparseSpMatDescr_t descrA, descrB, descrC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    
    hipsparseOperation_t opA         = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t opB         = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    
    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) )
    {
        // CUSPARSE context
        CHECK_CUSPARSE( hipsparseCreate( &handle ));
        CHECK_CUSPARSE( hipsparseSetStream( handle, queue->hip_stream() ));

        CHECK_CUSPARSE( hipsparseCreateCsr(&descrA, A.num_rows, A.num_cols, A.nnz,
                                          A.drow, A.dcol, A.dval,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_C_64F) );
        CHECK_CUSPARSE( hipsparseCreateCsr(&descrB, B.num_rows, B.num_cols, B.nnz,
                                          B.drow, B.dcol, B.dval,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_C_64F) );
        CHECK_CUSPARSE( hipsparseCreateCsr(&descrC, A.num_rows, B.num_cols, 0,
                                          NULL, NULL, NULL,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_C_64F) );

        // SpGEMM Computation
        hipsparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE( hipsparseSpGEMM_createDescr(&spgemmDesc) );

        // ask bufferSize1 bytes for external memory
        hipblasDoubleComplex alpha = MAGMA_Z_ONE, *beta = NULL; 
        CHECK_CUSPARSE( hipsparseSpGEMM_workEstimation(handle, opA, opB,
                                                      &alpha, descrA, descrB, beta, descrC,
                                                      HIPBLAS_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, NULL) );
        CHECK( hipMalloc((void**) &dBuffer1, bufferSize1) );

        // inspect the matrices A and B to understand the memory requirement for the next step
        CHECK_CUSPARSE( hipsparseSpGEMM_workEstimation(handle, opA, opB,
                                                      &alpha, descrA, descrB, beta, descrC,
                                                      HIPBLAS_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, dBuffer1) );

        // ask bufferSize2 bytes for external memory
        CHECK_CUSPARSE( hipsparseSpGEMM_compute(handle, opA, opB,
                                               &alpha, descrA, descrB, beta, descrC,
                                               HIPBLAS_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, NULL) );
        CHECK( hipMalloc((void**) &dBuffer2, bufferSize2) );

        // compute the intermediate product of A * B
        CHECK_CUSPARSE( hipsparseSpGEMM_compute(handle, opA, opB,
                                               &alpha, descrA, descrB, beta, descrC,
                                               HIPBLAS_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, dBuffer2) );

        // get matrix C non-zero entries C.nnz
        int64_t num_rows, num_cols, nnz;
        CHECK_CUSPARSE( hipsparseSpMatGetSize(descrC, &num_rows, &num_cols, &nnz) );
        C.num_rows = num_rows;
        C.num_cols = num_cols;
        C.nnz = nnz;
        
        // allocate matrix C
        CHECK( magma_index_malloc( &C.dcol, C.nnz ));
        CHECK( magma_zmalloc( &C.dval, C.nnz ));
        CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));
        
        // update matC with the new pointers
        CHECK_CUSPARSE( hipsparseCsrSetPointers(descrC, C.drow, C.dcol, C.dval) );

        // copy the final products to the matrix C
        CHECK_CUSPARSE( hipsparseSpGEMM_copy(handle, opA, opB,
                                            &alpha, descrA, descrB, beta, descrC,
                                            HIPBLAS_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) );

        // end CUSPARSE context
        magma_queue_sync( queue );
        CHECK( magma_zmtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( hipsparseSpGEMM_destroyDescr(spgemmDesc) );
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    CHECK_CUSPARSE( hipsparseDestroySpMat(descrA) );
    CHECK_CUSPARSE( hipsparseDestroySpMat(descrB) );
    CHECK_CUSPARSE( hipsparseDestroySpMat(descrC) );
    hipsparseDestroy( handle );
    magma_zmfree( &C, queue );
    return info;
    //============================================================================
#endif
 
}
