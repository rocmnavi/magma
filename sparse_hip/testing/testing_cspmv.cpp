/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from sparse_hip/testing/testing_zspmv.cpp, normal z -> c, Mon Jul 15 16:58:27 2024
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef MAGMA_WITH_MKL
    #include <mkl_spblas.h>

    #define PRECISION_c
    #if defined(PRECISION_z)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #elif defined(PRECISION_c)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #else
    #define MKL_ADDR(a) (a)
    #endif
#endif

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#if CUDA_VERSION >= 12000
  #define CUSPARSE_CSRMV_ALG2 CUSPARSE_SPMV_CSR_ALG2
  #define CUSPARSE_CSRMV_ALG1 CUSPARSE_SPMV_CSR_ALG1
  #define CUSPARSE_CSRMM_ALG1 CUSPARSE_SPMM_CSR_ALG1
#endif

#if CUDA_VERSION >= 11000
// todo: destroy descriptor and see if the original code descriptors have to be changed 
#define hipsparseCcsrmv(handle, op, rows, cols, nnz, alpha, descr, dval, drow, dcol, x, beta, y) \
    {                                                                                           \
        hipsparseSpMatDescr_t descrA;                                                            \
        hipsparseDnVecDescr_t descrX, descrY;                                                    \
        hipsparseCreateCsr(&descrA, rows, cols, nnz,                                             \
                          (void *)drow, (void *)dcol, (void *)dval,                             \
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,                               \
                          HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_C_64F);                                \
        hipsparseCreateDnVec(&descrX, cols, x, HIPBLAS_C_64F);                                      \
        hipsparseCreateDnVec(&descrY, rows, y, HIPBLAS_C_64F);                                      \
                                                                                                \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseSpMV_bufferSize(handle, op,                                                     \
                                (void *)alpha, descrA, descrX, (void *)beta,                    \
                                descrY, HIPBLAS_C_64F, CUSPARSE_CSRMV_ALG1, &bufsize);             \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseSpMV( handle, op,                                                               \
                      (void *)alpha, descrA, descrX, (void *)beta,                              \
                      descrY, HIPBLAS_C_64F, CUSPARSE_CSRMV_ALG1, buf);                            \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
        ;\
    }
#endif

#define PRECISION_c

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasComplex hipFloatComplex
#elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    magma_c_matrix hA={Magma_CSR}, hA_SELLP={Magma_CSR}, hA_ELL={Magma_CSR}, 
    dA={Magma_CSR}, dA_SELLP={Magma_CSR}, dA_ELL={Magma_CSR},
    hA_CSR5={Magma_CSR}, dA_CSR5={Magma_CSR};
    
    magma_c_matrix hx={Magma_CSR}, hy={Magma_CSR}, dx={Magma_CSR}, 
    dy={Magma_CSR}, hrefvec={Magma_CSR}, hcheck={Magma_CSR};
            
    hA_SELLP.blocksize = 32;
    hA_SELLP.alignment = 1;
    real_Double_t start, end, res, ref;
    real_Double_t elltime = 0.0, ellgflops = 0.0, mkltime = 0.0, mklgflops = 0.0, 
                  cuCSRtime = 0.0, cuCSRgflops = 0.0, 
                  cuHYBtime = 0.0, cuHYBgflops = 0.0, sellptime = 0.0, sellpgflops = 0.0, 
                  csr5time = 0.0, csr5gflops = 0.0;

    magmaFloatComplex c_one  = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex c_zero = MAGMA_C_MAKE(0.0, 0.0);
    
    float accuracy = 1e-8;
    
    #define PRECISION_c
    #if defined(PRECISION_c)
        accuracy = 1e-4;
    #endif
    #if defined(PRECISION_s)
        accuracy = 1e-4;
    #endif
    
    hipsparseMatDescr_t descrA=NULL;
    hipsparseHandle_t hipsparseHandle = NULL;
    hipsparseMatDescr_t descr = NULL;
    
    #ifdef MAGMA_WITH_MKL
        magma_int_t *pntre=NULL;
    #endif
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf( "\n%% #    usage: ./run_cspmv"
            " [ --blocksize %lld --alignment %lld (for SELLP) ] matrices\n\n",
            (long long) hA_SELLP.blocksize, (long long) hA_SELLP.alignment );

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_cm_5stencil(  laplace_size, &hA, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_c_csr_mtx( &hA,  argv[i], queue ));
        }

        printf( "\n%% # matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) hA.num_rows, (long long) hA.num_cols, (long long) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;

        // init CPU vectors
        TESTING_CHECK( magma_cvinit( &hx, Magma_CPU, hA.num_rows, 1, c_zero, queue ));
        TESTING_CHECK( magma_cvinit( &hy, Magma_CPU, hA.num_rows, 1, c_zero, queue ));

        // init DEV vectors
        TESTING_CHECK( magma_cvinit( &dx, Magma_DEV, hA.num_rows, 1, c_one, queue ));
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
       
        #ifdef MAGMA_WITH_MKL
            // calling MKL with CSR
            TESTING_CHECK( magma_imalloc_cpu( &pntre, hA.num_rows + 1 ) );
            pntre[0] = 0;
            for (j=0; j < hA.num_rows; j++ ) {
                pntre[j] = hA.row[j+1];
            }
             MKL_INT num_rows = hA.num_rows;
             MKL_INT num_cols = hA.num_cols;
             MKL_INT nnz = hA.nnz;

            MKL_INT *col;
            TESTING_CHECK( magma_malloc_cpu( (void**) &col, nnz * sizeof(MKL_INT) ));
            for( magma_int_t t=0; t < hA.nnz; ++t ) {
                col[ t ] = hA.col[ t ];
            }
            MKL_INT *row;
            TESTING_CHECK( magma_malloc_cpu( (void**) &row, num_rows * sizeof(MKL_INT) ));
            for( magma_int_t t=0; t < hA.num_rows; ++t ) {
                row[ t ] = hA.col[ t ];
            }
    
            start = magma_wtime();
            for (j=0; j < 200; j++ ) {
                mkl_ccsrmv( "N", &num_rows, &num_cols,
                            MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val),
                            col, row, pntre,
                                                    MKL_ADDR(hx.val),
                            MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
            }
            end = magma_wtime();
            mkltime = (end-start)/200;
            mklgflops = FLOPS*200/(end-start);
            printf( " \n%% > MKL  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/200, FLOPS*200/(end-start) );

            magma_free_cpu( row );
            magma_free_cpu( col );
            row = NULL;
            col = NULL;
        #endif // MAGMA_WITH_MKL


        // copy matrix to GPU
        TESTING_CHECK( magma_cmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));
        
        // warmup
        for (j=0; j < 200; j++) {
            TESTING_CHECK( magma_c_spmv( c_one, dA, dx, c_zero, dy, queue ));
        }

        // SpMV on GPU (CSR) -- this is the reference!
        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            TESTING_CHECK( magma_c_spmv( c_one, dA, dx, c_zero, dy, queue ));
        }
        end = magma_sync_wtime( queue );
        printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/200, FLOPS*200/(end-start) );
        
        magma_cmfree(&dA, queue );
        TESTING_CHECK( magma_cmtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue ));
        ref = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            ref = ref + MAGMA_C_ABS(hrefvec.val[k]);
        }

        // convert to ELL and copy to GPU
        TESTING_CHECK( magma_cmconvert(  hA, &hA_ELL, Magma_CSR, Magma_ELL, queue ));
        TESTING_CHECK( magma_cmtransfer( hA_ELL, &dA_ELL, Magma_CPU, Magma_DEV, queue ));
        magma_cmfree(&hA_ELL, queue );
        magma_cmfree( &dy, queue );
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        // SpMV on GPU (ELL)
        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            TESTING_CHECK( magma_c_spmv( c_one, dA_ELL, dx, c_zero, dy, queue ));
        }
        end = magma_sync_wtime( queue );
        magma_cmfree(&dA_ELL, queue );
        TESTING_CHECK( magma_cmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            res = res + MAGMA_C_ABS(hcheck.val[k] - hrefvec.val[k]);
        }
        res /= ref;
        if ( res < accuracy ) {
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                (end-start)/200, FLOPS*200/(end-start) );
            printf("%% |x-y|_F/|y| = %8.2e.  Tester spmv ELL:  ok\n", res);
            elltime = (end-start)/200;
            ellgflops = FLOPS*200/(end-start);
        } else {
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                (end-start)/200, 0.0 );
            printf("%% |x-y|_F/|y| = %8.2e.  Tester spmv ELL:  failed\n", res);
            elltime = NAN;
            ellgflops = NAN;
        }
        magma_cmfree( &hcheck, queue );

        // convert to SELLP and copy to GPU
        TESTING_CHECK( magma_cmconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue ));
        TESTING_CHECK( magma_cmtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue ));
        magma_cmfree(&hA_SELLP, queue );
        magma_cmfree( &dy, queue );
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            TESTING_CHECK( magma_c_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue ));
        }
        end = magma_sync_wtime( queue );
        TESTING_CHECK( magma_cmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            res = res + MAGMA_C_ABS(hcheck.val[k] - hrefvec.val[k]);
        }
        //res /= ref;
        res = ref == 0 ? res : res / ref;
        if ( res < accuracy ) {
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                (end-start)/200, FLOPS*200/(end-start) );
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv SELL-P:  ok\n", res);
            sellptime = (end-start)/200;
            sellpgflops = FLOPS*200/(end-start);
        } else{
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                (end-start)/200, 0.0);
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv SELL-P:  failed\n", res);
            sellptime = NAN;
            sellpgflops = NAN;
        }
        magma_cmfree( &hcheck, queue );

        magma_cmfree(&dA_SELLP, queue );

        // convert to CSR5 and copy to GPU
        TESTING_CHECK( magma_cmconvert(  hA, &hA_CSR5, Magma_CSR, Magma_CSR5, queue ));
        TESTING_CHECK( magma_cmtransfer( hA_CSR5, &dA_CSR5, Magma_CPU, Magma_DEV, queue ));
        magma_cmfree(&hA_CSR5, queue );
        magma_cmfree( &dy, queue );
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        // SpMV on GPU (CSR5)
        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            info = magma_c_spmv( c_one, dA_CSR5, dx, c_zero, dy, queue );
        }
        if( info != 0 ){
            printf("%% error: CSR5 not supported\n");    
        }
        end = magma_sync_wtime( queue );
        TESTING_CHECK( magma_cmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            res = res + MAGMA_C_ABS(hcheck.val[k] - hrefvec.val[k]);
        }
        //res /= ref
        res = ref == 0 ? res : res / ref;
        if ( res < accuracy ) {
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (CSR5).\n",
                (end-start)/200, FLOPS*200/(end-start) );
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv CSR5:  ok\n", res);
            csr5time = (end-start)/200;
            csr5gflops = FLOPS*200/(end-start);
        } else{
            printf( "%% > MAGMA: %.2e seconds %.2e GFLOP/s    (CSR5).\n",
                (end-start)/200, 0.0);
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv CSR5:  failed\n", res);
            csr5time = NAN;
            csr5gflops = NAN;
        }
        magma_cmfree( &hcheck, queue );

        magma_cmfree(&dA_CSR5, queue );


        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context
        hipsparseHandle = magma_queue_get_hipsparse_handle( queue );
        TESTING_CHECK( hipsparseCreateMatDescr( &descr ));
             
        TESTING_CHECK( hipsparseSetMatType( descr, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        TESTING_CHECK( hipsparseSetMatIndexBase( descr, HIPSPARSE_INDEX_BASE_ZERO ));
        magmaFloatComplex alpha = c_one;
        magmaFloatComplex beta = c_zero;
        magma_cmfree( &dy, queue );
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));

        // copy matrix to GPU
        TESTING_CHECK( magma_cmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));

        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            hipsparseCcsrmv(hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                               hA.num_rows, hA.num_cols, hA.nnz, (hipblasComplex*)&alpha, descr,
                               (hipblasComplex*)dA.dval, dA.drow, dA.dcol, (hipblasComplex*)dx.dval, 
                               (hipblasComplex*)&beta, (hipblasComplex*)dy.dval);
        }
        end = magma_sync_wtime( queue );

        TESTING_CHECK( magma_cmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            res = res + MAGMA_C_ABS(hcheck.val[k] - hrefvec.val[k]);
        }
        //res /= ref;
        res = ref == 0 ? res : res / ref;
        if ( res < accuracy ) {
            printf( "%% > cuSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                (end-start)/200, FLOPS*200/(end-start) );
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv cuSPARSE CSR:  ok\n", res);
            cuCSRtime = (end-start)/200;
            cuCSRgflops = FLOPS*200/(end-start);
        } else{
            printf( "%% > cuSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                (end-start)/200, 0.0);
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv cuSPARSE CSR:  failed\n", res);
            cuCSRtime = NAN;
            cuCSRgflops = NAN;
        }
        magma_cmfree( &hcheck, queue );
        magma_cmfree( &dy, queue );


#if defined(MAGMA_HAVE_CUDA) && CUDA_VERSION < 11000
        // Test hybrid matix format for CUDA before version 11 ===
        hipsparseHybMat_t hybA=NULL;
        TESTING_CHECK( hipsparseCreateMatDescr( &descrA ));
        hipsparseCreateHybMat( &hybA );
        TESTING_CHECK( magma_cvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        hipsparseCcsr2hyb(hipsparseHandle,  hA.num_rows, hA.num_cols,
                        descrA, dA.dval, dA.drow, dA.dcol,
                        hybA, 0, HIPSPARSE_HYB_PARTITION_AUTO);

        start = magma_sync_wtime( queue );
        for (j=0; j < 200; j++) {
            TESTING_CHECK( hipsparseChybmv( hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                       &alpha, descrA, hybA,
                       dx.dval, &beta, dy.dval) );
        }
        end = magma_sync_wtime( queue );

        TESTING_CHECK( magma_cmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ){
            res = res + MAGMA_C_ABS(hcheck.val[k] - hrefvec.val[k]);
        }
        //res /= ref;
        res = ref == 0 ? res : res / ref;
        if ( res < accuracy ) {
            printf( "%% > cuSPARSE: %.2e seconds %.2e GFLOP/s    (HYB).\n",
                (end-start)/200, FLOPS*200/(end-start) );
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv cuSPARSE HYB:  ok\n", res);
            cuHYBtime = (end-start)/200;
            cuHYBgflops = FLOPS*200/(end-start);
        } else{
            printf( "%% > cuSPARSE: %.2e seconds %.2e GFLOP/s    (HYB).\n",
                (end-start)/200, 0.0);
            printf("%% |x-y|_F/|y| = %8.2e Tester spmv cuSPARSE HYB:  failed\n", res);
            cuHYBtime = NAN;
            cuHYBgflops = NAN;
        }
        magma_cmfree( &hcheck, queue );

        hipsparseDestroyMatDescr( descrA );
        hipsparseDestroyHybMat( hybA );
        descrA=NULL;
        hybA=NULL;
#endif // end test for HYB matrix format

        hipsparseDestroyMatDescr( descr  );
        descr = NULL;
        
        // print everything in matlab-readable output
        // cuSPARSE-CSR cuSPARSE-HYB  SELLP  CSR5
        // runtime performance 
        // printf("%% MKL cuSPARSE-CSR cuSPARSE-HYB  ELL SELLP  CSR5\n");
        // printf("%% runtime performance (GFLOP/s)\n");
        // printf("data = [\n");
        printf(" MKL (sec GFlop/s)   cuCSR (s GFlop/s)   cuHYB (s GFlop/s)   ELL (sec GFlop/s)   Sell (s  GFlop/s)   CSR5 (s GFlop/s)\n");
        printf("=====================================================================================================================\n");
        printf(" %.2e %.2e   %.2e %.2e   %.2e %.2e   %.2e %.2e   %.2e %.2e   %.2e %.2e\n",
                 mkltime, mklgflops, cuCSRtime, cuCSRgflops, cuHYBtime, cuHYBgflops, 
                 elltime, ellgflops, sellptime, sellpgflops, csr5time, csr5gflops);

        // free CPU memory
        magma_cmfree( &hA, queue );
        magma_cmfree( &hx, queue );
        magma_cmfree( &hy, queue );
        magma_cmfree( &hrefvec, queue );
        // free GPU memory
        magma_cmfree( &dA, queue );
        magma_cmfree( &dx, queue );
        magma_cmfree( &dy, queue );

        printf("\n\n");

        #ifdef MAGMA_WITH_MKL
            magma_free_cpu( pntre );
        #endif
        
        i++;
    }
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
