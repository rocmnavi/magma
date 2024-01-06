/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from sparse/blas/magma_zcuspmm.cpp, normal z -> s, Fri Aug 25 13:19:05 2023
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

#if CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
#define cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA, drowA, dcolA,        \
                            descrB, nnzB, drowB, dcolB, descrC, drowC, nnzTotal )               \
    {                                                                                           \
        cusparseMatDescr_t descrD;                                                              \
        int nnzD, *drowD, *dcolD;                                                               \
        csrgemm2Info_t linfo = NULL;                                                            \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        float alpha = MAGMA_S_ONE, *beta = NULL;                                      \
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);                             \
        cusparseCreateCsrgemm2Info(&linfo);                                                     \
        cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha,                                \
                                        descrA, nnzA, drowA, dcolA,                             \
                                        descrB, nnzB, drowB, dcolB,                             \
                                        beta,                                                   \
                                        descrD, nnzD, drowD, dcolD,                             \
                                        linfo, &bufsize);                                       \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseXcsrgemm2Nnz(handle, m, n, k,                                                   \
                             descrA, nnzA, drowA, dcolA,                                        \
                             descrB, nnzB, drowB, dcolB,                                        \
                             descrD, nnzD, drowD, dcolD,                                        \
                             descrC, drowC, nnzTotal,                                           \
                             linfo, buf);                                                       \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

// todo: info and buf are passed from the above function
// also at the end destroy info: cusparseDestroyCsrgemm2Info(info);
#define cusparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, dvalA, drowA, dcolA,    \
                         descrB, nnzB, dvalB, drowB, dcolB, descrC, dvalC, drowC, dcolC )       \
    {                                                                                           \
        cusparseMatDescr_t descrD;                                                              \
        int nnzD, *drowD, *dcolD;                                                               \
        float *dvalD, alpha = MAGMA_S_ONE, *beta = NULL;                              \
        void *buf;                                                                              \
        csrgemm2Info_t linfo = NULL;                                                            \
        printf("cusparseScsrgemm bufsize = %d\n", -1);                                          \
        cusparseScsrgemm2(handle, m, n, k, &alpha,                                              \
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
    A           magma_s_matrix
                input matrix

    @param[in]
    B           magma_s_matrix
                input matrix

    @param[out]
    AB          magma_s_matrix*
                output matrix AB = A * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sblas
    ********************************************************************/

extern "C" magma_int_t
magma_scuspmm(
    magma_s_matrix A, magma_s_matrix B,
    magma_s_matrix *AB,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    
    magma_s_matrix C={Magma_CSR};
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
    
    cusparseHandle_t handle=NULL;
    cusparseMatDescr_t descrA=NULL;
    cusparseMatDescr_t descrB=NULL;
    cusparseMatDescr_t descrC=NULL;
    
    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) )
    {
        // CUSPARSE context /
        CHECK_CUSPARSE( cusparseCreate( &handle ));
        CHECK_CUSPARSE( cusparseSetStream( handle, queue->cuda_stream() ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrB ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrC ));
        CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatType( descrB, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatType( descrC, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrB, CUSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descrC, CUSPARSE_INDEX_BASE_ZERO ));

        // nnzTotalDevHostPtr points to host memory
        magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &C.nnz;
        CHECK_CUSPARSE( cusparseSetPointerMode( handle, CUSPARSE_POINTER_MODE_HOST ));
        CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));
        cusparseXcsrgemmNnz( handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
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
        CHECK( magma_smalloc( &C.dval, C.nnz ));
        #ifdef MAGMA_HAVE_HIP
        hipsparseScsrgemm( handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                          A.num_rows, B.num_cols, A.num_cols,
                          descrA, A.nnz,
                          (const float*)A.dval, A.drow, A.dcol,
                          descrB, B.nnz,
                          (const float*)B.dval, B.drow, B.dcol,
                          descrC,
                          (float*)C.dval, C.drow, C.dcol );

        #else
        cusparseScsrgemm( handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
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
        CHECK( magma_smtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
cleanup:
    cusparseDestroyMatDescr( descrA );
    cusparseDestroyMatDescr( descrB );
    cusparseDestroyMatDescr( descrC );
    cusparseDestroy( handle );
    magma_smfree( &C, queue );
    return info;

#else
    // ================================================================================
    magma_index_t base_t, nnz_t, baseC;

    cusparseHandle_t handle=NULL;
    cusparseSpMatDescr_t descrA, descrB, descrC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    
    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) )
    {
        // CUSPARSE context
        CHECK_CUSPARSE( cusparseCreate( &handle ));
        CHECK_CUSPARSE( cusparseSetStream( handle, queue->cuda_stream() ));

        CHECK_CUSPARSE( cusparseCreateCsr(&descrA, A.num_rows, A.num_cols, A.nnz,
                                          A.drow, A.dcol, A.dval,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
        CHECK_CUSPARSE( cusparseCreateCsr(&descrB, B.num_rows, B.num_cols, B.nnz,
                                          B.drow, B.dcol, B.dval,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
        CHECK_CUSPARSE( cusparseCreateCsr(&descrC, A.num_rows, B.num_cols, 0,
                                          NULL, NULL, NULL,
                                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

        // SpGEMM Computation
        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) );

        // ask bufferSize1 bytes for external memory
        float alpha = MAGMA_S_ONE, *beta = NULL; 
        CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                      &alpha, descrA, descrB, beta, descrC,
                                                      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, NULL) );
        CHECK( cudaMalloc((void**) &dBuffer1, bufferSize1) );

        // inspect the matrices A and B to understand the memory requirement for the next step
        CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, opA, opB,
                                                      &alpha, descrA, descrB, beta, descrC,
                                                      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, dBuffer1) );

        // ask bufferSize2 bytes for external memory
        CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                               &alpha, descrA, descrB, beta, descrC,
                                               CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, NULL) );
        CHECK( cudaMalloc((void**) &dBuffer2, bufferSize2) );

        // compute the intermediate product of A * B
        CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                               &alpha, descrA, descrB, beta, descrC,
                                               CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, dBuffer2) );

        // get matrix C non-zero entries C.nnz
        int64_t num_rows, num_cols, nnz;
        CHECK_CUSPARSE( cusparseSpMatGetSize(descrC, &num_rows, &num_cols, &nnz) );
        C.num_rows = num_rows;
        C.num_cols = num_cols;
        C.nnz = nnz;
        
        // allocate matrix C
        CHECK( magma_index_malloc( &C.dcol, C.nnz ));
        CHECK( magma_smalloc( &C.dval, C.nnz ));
        CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));
        
        // update matC with the new pointers
        CHECK_CUSPARSE( cusparseCsrSetPointers(descrC, C.drow, C.dcol, C.dval) );

        // copy the final products to the matrix C
        CHECK_CUSPARSE( cusparseSpGEMM_copy(handle, opA, opB,
                                            &alpha, descrA, descrB, beta, descrC,
                                            CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) );

        // end CUSPARSE context
        magma_queue_sync( queue );
        CHECK( magma_smtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) );
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    CHECK_CUSPARSE( cusparseDestroySpMat(descrA) );
    CHECK_CUSPARSE( cusparseDestroySpMat(descrB) );
    CHECK_CUSPARSE( cusparseDestroySpMat(descrC) );
    cusparseDestroy( handle );
    magma_smfree( &C, queue );
    return info;
    //============================================================================
#endif
 
}
