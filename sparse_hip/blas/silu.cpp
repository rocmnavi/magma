/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Hartwig Anzt

       @generated from sparse_hip/blas/zilu.cpp, normal z -> s, Fri Aug 25 13:17:51 2023
*/
#include "magmasparse_internal.h"
#include <hip/hip_runtime.h>  // for CUDA_VERSION

#include "magma_trisolve.h"

#define PRECISION_s

/* For hipSPARSE, they use a separate real type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define float float
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

#if CUDA_VERSION >= 12000
   #define hipsparseCreateCsrsm2Info(info)
   #define hipsparseDestroyCsrsm2Info(info)
#endif

// todo: make it spacific
#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define hipsparseCreateSolveAnalysisInfo(info) hipsparseCreateCsrsm2Info(info) 
#else
#define hipsparseCreateSolveAnalysisInfo(info)                                                   \
        CHECK_CUSPARSE( hipsparseCreateSolveAnalysisInfo( info ))
#endif

#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define hipsparseDestroySolveAnalysisInfo(info) hipsparseDestroyCsrsm2Info(info)
#endif

// todo: check the info and linfo if we have to give it back; free memory? 
#if CUDA_VERSION >= 11000
#define hipsparseScsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        float alpha = MAGMA_S_ONE;                                                 \
        float *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_GENERAL );                             \
        hipsparseScsrsm2_bufferSizeExt(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,          \
                                      rows, 1, nnz, (const float *)&alpha,            \
                                      descrA, dval, drow, dcol,                                 \
                                      B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseScsrsm2_analysis(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const float *)&alpha,                 \
                                 descrA, dval, drow, dcol,                                      \
                                 B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

#elif defined(MAGMA_HAVE_HIP)
#define hipsparseScsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        float alpha = MAGMA_S_ONE;                                                 \
        float *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseScsrsm2_bufferSizeExt(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,         \
                                      rows, 1, nnz, (const float *)&alpha,           \
                                      descrA, (const float *)dval, (const int *)drow, (const int *)dcol,  \
                                      (const float *)B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseScsrsm2_analysis(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const float *)&alpha,                 \
                                 descrA, (const float *)dval, drow, dcol,            \
                                 B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }



#endif

#if CUDA_VERSION >= 11000
#define hipsparseScsr2csc(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,     \
                         action, base)                                                          \
    {                                                                                           \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseCsr2cscEx2_bufferSize(handle, cols, rows, nnz, dval, drow, dcol, prdval,        \
                                      prdcol, prdrow, HIPBLAS_C_64F, action, base,                 \
                                      CUSPARSE_CSR2CSC_ALG1, &bufsize);                         \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseCsr2cscEx2(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,   \
                           HIPBLAS_C_64F, action, base, CUSPARSE_CSR2CSC_ALG1, buf);               \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

// todo: info is passed from analysis; to change info with this linfo & remove linfo from here
#if CUDA_VERSION >= 11000
#define hipsparseScsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        hipsparseCreateCsric02Info(&linfo);                                                      \
        hipsparseScsric02_bufferSize(handle, rows, nnz, descrA, dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseScsric02_analysis(handle, rows, nnz, descrA, dval, drow, dcol, linfo,           \
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (HIPSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            hipsparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        hipsparseScsric02(handle, rows, nnz, descrA, dval, drow, dcol, linfo,                    \
                         HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#elif defined(MAGMA_HAVE_HIP)
#define hipsparseScsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        hipsparseCreateCsric02Info(&linfo);                                                      \
        hipsparseScsric02_bufferSize(handle, rows, nnz, descrA, (float*)dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseScsric02_analysis(handle, rows, nnz, descrA, (float*)dval, drow, dcol, linfo,           \
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (HIPSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            hipsparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        hipsparseScsric02(handle, rows, nnz, descrA, (float*)dval, drow, dcol, linfo,                    \
                         HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    } 
#else
#define hipsparseScsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    CHECK_CUSPARSE( hipsparseScsric0(handle, op, rows, descrA, dval, drow, dcol, info ))
#endif

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_scumilusetup(
    magma_s_matrix A,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    hipsparseHandle_t hipsparseHandle=NULL;
    hipsparseMatDescr_t descrA=NULL;
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    csrilu02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    // magma_sprint_matrix(A, queue );
    // copy matrix into preconditioner parameter
    magma_s_matrix hA={Magma_CSR}, hACSR={Magma_CSR};
    magma_s_matrix hL={Magma_CSR}, hU={Magma_CSR};
    CHECK( magma_smtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_smconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
        magma_s_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
        CHECK( magma_ssymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
        magma_smfree(&hAL, queue);
        magma_smfree(&hAUt, queue);
    }

    CHECK( magma_smtransfer(hACSR, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    magma_smfree( &hA, queue );
    magma_smfree( &hACSR, queue );

    // CUSPARSE context //
    CHECK_CUSPARSE( hipsparseCreate( &hipsparseHandle ));
    CHECK_CUSPARSE( hipsparseSetStream( hipsparseHandle, queue->hip_stream() ));
    CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( hipsparseSetMatDiagType( descrA, HIPSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, HIPSPARSE_INDEX_BASE_ZERO ));
    hipsparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );

    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_sdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 7000 
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( hipsparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    hipsparseScsrilu02_bufferSize( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseScsrilu02_analysis( hipsparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( hipsparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( hipsparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    hipsparseScsrilu02( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#elif defined(MAGMA_HAVE_HIP)

    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( hipsparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    hipsparseScsrilu02_bufferSize( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (float*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseScsrilu02_analysis( hipsparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            (float*)precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( hipsparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( hipsparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    hipsparseScsrilu02( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (float*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#else
    // this version contains the bug but is needed for backward compability
    hipsparseScsrsm_analysis( hipsparseHandle,
                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    CHECK_CUSPARSE( hipsparseScsrilu0( hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                      precond->M.num_rows, descrA,
                      precond->M.dval,
                      precond->M.drow,
                      precond->M.dcol,
                      precond->cuinfoILU ));
#endif

    CHECK( magma_smtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));

    hL.diagorder_type = Magma_UNITY;
    CHECK( magma_smconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
    hU.diagorder_type = Magma_VALUE;
    CHECK( magma_smconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
    CHECK( magma_smtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_smtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
    
    // malloc aux space for sync-free sptrsv 
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree_bak), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree_bak), precond->M.num_rows ));

    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_strisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
        CHECK(magma_strisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
            magma_smfree(&hL, queue );
            magma_smfree(&hU, queue );
            magma_smtransfer( precond->L, &hL, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE
            #ifdef MAGMA_HAVE_HIP
            hipsparseScsr2csc(hipsparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             (float*)hL.dval, hL.drow, hL.dcol, 
                             (float*)precond->L.dval, precond->L.dcol, precond->L.drow,
                             HIPSPARSE_ACTION_NUMERIC,
                             HIPSPARSE_INDEX_BASE_ZERO);
            #else
            hipsparseScsr2csc(hipsparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             hL.dval, hL.drow, hL.dcol, 
                             precond->L.dval, precond->L.dcol, precond->L.drow,
                             HIPSPARSE_ACTION_NUMERIC,
                             HIPSPARSE_INDEX_BASE_ZERO);

            #endif

            magma_smtransfer( precond->U, &hU, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE

            #ifdef MAGMA_HAVE_HIP
            hipsparseScsr2csc(hipsparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             (float*)hU.dval, hU.drow, hU.dcol, 
                             (float*)precond->U.dval, precond->U.dcol, precond->U.drow,
                             HIPSPARSE_ACTION_NUMERIC,
                             HIPSPARSE_INDEX_BASE_ZERO);
            #else
            hipsparseScsr2csc(hipsparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             hU.dval, hU.drow, hU.dcol, 
                             precond->U.dval, precond->U.dcol, precond->U.drow,
                             HIPSPARSE_ACTION_NUMERIC,
                             HIPSPARSE_INDEX_BASE_ZERO);
            #endif

            // set this to be CSC
            precond->U.storage_type = Magma_CSC;
            precond->L.storage_type = Magma_CSC;
            
            // analysis sparsity structures of L and U
            magma_sgecscsyncfreetrsm_analysis(precond->L.num_rows, 
                precond->L.nnz, precond->L.dval, 
                precond->L.drow, precond->L.dcol, 
                precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
                queue);
            magma_sgecscsyncfreetrsm_analysis(precond->U.num_rows, 
                precond->U.nnz, precond->U.dval, 
                precond->U.drow, precond->U.dcol, 
                precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
                queue);

            magma_smfree(&hL, queue );
            magma_smfree(&hU, queue );
    } else {
        //prepare for iterative solves
        
        // extract the diagonal of L into precond->d
        CHECK( magma_sjacobisetup_diagscal( precond->L, &precond->d, queue ));
        // precond->d.memory_location = Magma_DEV;
        CHECK( magma_svinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_S_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_sjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        // precond->d2.memory_location = Magma_DEV;
        CHECK( magma_svinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_S_ZERO, queue ));
    }

    
cleanup:
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    magma_free( pBuffer );
    hipsparseDestroyCsrilu02Info( info_M );
#endif
    hipsparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    hipsparseDestroyMatDescr( descrA );
    hipsparseDestroy( hipsparseHandle );
    magma_smfree( &hA, queue );
    magma_smfree( &hACSR, queue );
    magma_smfree(&hA, queue );
    magma_smfree(&hL, queue );
    magma_smfree(&hU, queue );

    return info;
}



/**
    Purpose
    -------

    Prepares the ILU transpose preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_scumilusetup_transpose(
    magma_s_matrix A,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_s_matrix Ah1={Magma_CSR}, Ah2={Magma_CSR};

    // transpose the matrix
    magma_smtransfer( precond->L, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_smconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransposeconjugate( Ah2, &Ah1, queue );
    magma_smfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_smconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransfer( Ah2, &(precond->LT), Magma_CPU, Magma_DEV, queue );
    magma_smfree(&Ah2, queue );
    
    magma_smtransfer( precond->U, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_smconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransposeconjugate( Ah2, &Ah1, queue );
    magma_smfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_smconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransfer( Ah2, &(precond->UT), Magma_CPU, Magma_DEV, queue );
    magma_smfree(&Ah2, queue );
   
    CHECK(magma_strisolve_analysis(precond->LT, &precond->cuinfoLT, true, false, false, queue));
    CHECK(magma_strisolve_analysis(precond->UT, &precond->cuinfoUT, false, false, false, queue));

cleanup:
    magma_smfree(&Ah1, queue );
    magma_smfree(&Ah2, queue );

    return info;
}



/**
    Purpose
    -------

    Prepares the ILU triangular solves via cuSPARSE using an ILU factorization
    matrix stored either in precond->M or on the device as
    precond->L and precond->U.

    Arguments
    ---------

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_scumilugeneratesolverinfo(
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_s_matrix hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR};
    
    if (precond->L.memory_location != Magma_DEV ){
        CHECK( magma_smtransfer( precond->M, &hA,
        precond->M.memory_location, Magma_CPU, queue ));

        hL.diagorder_type = Magma_UNITY;
        CHECK( magma_smconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
        hU.diagorder_type = Magma_VALUE;
        CHECK( magma_smconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
        CHECK( magma_smtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_smtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
        
        magma_smfree(&hA, queue );
        magma_smfree(&hL, queue );
        magma_smfree(&hU, queue );
    }
    
    CHECK(magma_strisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_strisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    
    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK( magma_sjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_svinit( &precond->work1, Magma_DEV, precond->U.num_rows, 1, MAGMA_S_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_sjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        CHECK( magma_svinit( &precond->work2, Magma_DEV, precond->U.num_rows, 1, MAGMA_S_ZERO, queue ));
    }
    
cleanup:     
    return info;
}


/**
    Purpose
    -------

    Performs the left triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_sapplycumilu_l(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_strisolve(precond->L, precond->cuinfoL, false, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_sgecscsyncfreetrsm_solve( precond->L.num_rows,
            precond->L.nnz, 
            precond->L.dval, precond->L.drow, precond->L.dcol, 
            precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
            x->dval, b.dval, 0, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD
            1, // rhs
            queue );
    }
       

cleanup:
    return info;
}



/**
    Purpose
    -------

    Performs the left triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
   
extern "C" magma_int_t
magma_sapplycumilu_l_transpose(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    CHECK(magma_strisolve(precond->LT, precond->cuinfoLT, true, false, false, b, *x, queue));
    
    

cleanup:
    return info;
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_sapplycumilu_r(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_strisolve(precond->U, precond->cuinfoU, true, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_sgecscsyncfreetrsm_solve( precond->U.num_rows,
            precond->U.nnz,
            precond->U.dval, precond->U.drow, precond->U.dcol, 
            precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
            x->dval, b.dval, 1, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_BACKWARD
            1, // rhs
            queue );
    }
    
    

cleanup:
    return info; 
}


/**
    Purpose
    -------

    Performs the right triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/

extern "C" magma_int_t
magma_sapplycumilu_r_transpose(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    CHECK(magma_strisolve(precond->UT, precond->cuinfoUT, false, false, false, b, *x, queue));
    
cleanup:
    return info; 
}


/**
    Purpose
    -------

    Prepares the IC preconditioner via cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_shepr
*******************************************************************************/

extern "C" magma_int_t
magma_scumiccsetup(
    magma_s_matrix A,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    hipsparseHandle_t hipsparseHandle=NULL;
    hipsparseMatDescr_t descrA=NULL;
    hipsparseMatDescr_t descrL=NULL;
    hipsparseMatDescr_t descrU=NULL;
#if CUDA_VERSION >= 7000
    csric02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    magma_s_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, U={Magma_CSR};
    CHECK( magma_smtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    U.diagorder_type = Magma_VALUE;
    CHECK( magma_smconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
            magma_s_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
            CHECK( magma_ssymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
            magma_smfree(&hAL, queue);
            magma_smfree(&hAUt, queue);
    }

    CHECK( magma_smconvert( hACSR, &U, Magma_CSR, Magma_CSRL, queue ));
    magma_smfree( &hACSR, queue );
    CHECK( magma_smtransfer(U, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    // CUSPARSE context //
    CHECK_CUSPARSE( hipsparseCreate( &hipsparseHandle ));
    CHECK_CUSPARSE( hipsparseSetStream( hipsparseHandle, queue->hip_stream() ));
    CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrA ));
    hipsparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );
    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_sdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 12000
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( hipsparseCreateCsric02Info(&info_M) );
    CHECK_CUSPARSE( hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, HIPSPARSE_INDEX_BASE_ZERO ));
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    hipsparseScsric02_bufferSize( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseScsric02_analysis( hipsparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    CHECK_CUSPARSE( hipsparseXcsric02_zeroPivot( hipsparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsric02_zeroPivot( hipsparseHandle, info_M, &structural_zero ) );

    CHECK_CUSPARSE(
    hipsparseScsric02( hipsparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );    

#else
    // this version contains the bug but is needed for backward compability
    CHECK_CUSPARSE( hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_SYMMETRIC ));
    CHECK_CUSPARSE( hipsparseSetMatDiagType( descrA, HIPSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, HIPSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( hipsparseSetMatFillMode( descrA, HIPSPARSE_FILL_MODE_LOWER ));
    
    // todo: Zcsric0 needs different analysis (hipsparseScsric02_analysis)
    hipsparseScsrsm_analysis( hipsparseHandle,
                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    hipsparseScsric0( hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                     precond->M.num_rows, precond->M.nnz, descrA,
                     precond->M.dval,
                     precond->M.drow,
                     precond->M.dcol,
                     precond->cuinfoILU );
#endif

    CHECK( magma_smtransfer( precond->M, &precond->L, 
        Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_smtranspose(precond->M, &precond->U, queue ));

    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_scumicgeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_sjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_svinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_S_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_sjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_svinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_S_ZERO, queue));
    }



/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_smtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    magma_s_matrix hD, hR, hAt

    CHECK( magma_scsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_smtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_smtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_smfree(&hD, queue );
    magma_smfree(&hR, queue );

    CHECK( magma_s_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_scsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_smtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_smtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_smfree(&hD, queue );
    magma_smfree(&hR, queue );
    magma_smfree(&hA, queue );
    magma_smfree(&hAt, queue );
*/

cleanup:
#if CUDA_VERSION >= 7000
    magma_free( pBuffer );
    hipsparseDestroyCsric02Info( info_M );
#endif
    hipsparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    hipsparseDestroyMatDescr( descrL );
    hipsparseDestroyMatDescr( descrU );
    hipsparseDestroyMatDescr( descrA );
    hipsparseDestroy( hipsparseHandle );
    magma_smfree(&U, queue );
    magma_smfree(&hA, queue );

    return info;
}


/**
    Purpose
    -------

    Prepares the IC preconditioner solverinfo via cuSPARSE for a triangular
    matrix present on the device in precond->M.

    Arguments
    ---------
    
    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_shepr
    ********************************************************************/

extern "C" magma_int_t
magma_scumicgeneratesolverinfo(
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    CHECK(magma_strisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_strisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
    

/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_smtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    CHECK( magma_scsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_smtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_smtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_smfree(&hD, queue );
    magma_smfree(&hR, queue );

    CHECK( magma_s_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_scsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_smtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_smtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_smfree(&hD, queue );
    magma_smfree(&hR, queue );
    magma_smfree(&hA, queue );
    magma_smfree(&hAt, queue );
*/

cleanup:
    return info;
}



/**
    Purpose
    -------

    Performs the left triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_shepr
    ********************************************************************/

extern "C" magma_int_t
magma_sapplycumicc_l(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    CHECK(magma_strisolve(precond->M, precond->cuinfoL, false, false, false, b, *x, queue));

cleanup:
    return info; 
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[in,out]
    x           magma_s_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_shepr
    ********************************************************************/

extern "C" magma_int_t
magma_sapplycumicc_r(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0);

    CHECK(magma_strisolve(precond->M, precond->cuinfoU, false, false, true, b, *x, queue));
    
    

cleanup:
    return info; 
}



/**
    Purpose
    -------

    Performs the left triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[out]
    x           magma_s_matrix*
                vector to precondition

    @param[in]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sapplyiteric_l(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t dofs = precond->L.num_rows;
    magma_s_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_sjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));
    // Jacobi iterator
    CHECK( magma_sjacobiiter_precond( precond->L, x, &jacobiiter_par, precond , queue ));

cleanup:
    return info;
}


/**
    Purpose
    -------

    Performs the right triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                RHS

    @param[out]
    x           magma_s_matrix*
                vector to precondition

    @param[in]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sapplyiteric_r(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t dofs = precond->U.num_rows;
    magma_s_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_sjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));

    // Jacobi iterator
    CHECK( magma_sjacobiiter_precond( precond->U, x, &jacobiiter_par, precond , queue ));
    
cleanup:
    return info;
}

