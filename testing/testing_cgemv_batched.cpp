/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from testing/testing_zgemv_batched.cpp, normal z -> c, Fri Aug 25 13:17:38 2023
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgemv_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, device_perf, device_time, cpu_perf, cpu_time;
    float          error, magma_error, device_error, normalize, work[1];
    magma_int_t M, N, Xm, Ym, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaFloatComplex *h_A, *h_X, *h_Y, *h_Ymagma, *h_Ydevice;
    magmaFloatComplex *d_A, *d_X, *d_Y;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( -0.48,  0.38 );
    magmaFloatComplex **d_A_array = NULL;
    magmaFloatComplex **d_X_array = NULL;
    magmaFloatComplex **d_Y_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;

    float *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Xnorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Ynorm, batchCount ));

    // See testing_cgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;

    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   %s Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error   %s error\n", g_platform_str, g_platform_str);
    printf("%%===================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_CGEMV( M, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            }
            else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N*batchCount;
            sizeX = incx*Xm*batchCount;
            sizeY = incy*Ym*batchCount;

            TESTING_CHECK( magma_cmalloc_cpu( &h_A,  sizeA ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_X,  sizeX ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Y,  sizeY  ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Ymagma,  sizeY  ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Ydevice, sizeY  ));

            TESTING_CHECK( magma_cmalloc( &d_A, ldda*N*batchCount ));
            TESTING_CHECK( magma_cmalloc( &d_X, sizeX ));
            TESTING_CHECK( magma_cmalloc( &d_Y, sizeY ));

            TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_X_array, batchCount * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_Y_array, batchCount * sizeof(magmaFloatComplex*) ));

            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_clarnv( &ione, ISEED, &sizeY, h_Y );

            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_clange( "F", &M, &N,     &h_A[s*lda*N],   &lda,  work );
                Xnorm[s] = lapackf77_clange( "F", &ione, &Xm, &h_X[s*Xm*incx], &incx, work );
                Ynorm[s] = lapackf77_clange( "F", &ione, &Ym, &h_Y[s*Ym*incy], &incy, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_csetmatrix( M, N*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_csetvector( Xm*batchCount, h_X, incx, d_X, incx, opts.queue );
            magma_csetvector( Ym*batchCount, h_Y, incy, d_Y, incy, opts.queue );

            magma_cset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_cset_pointer( d_X_array, d_X, 1, 0, 0, incx*Xm, batchCount, opts.queue );
            magma_cset_pointer( d_Y_array, d_Y, 1, 0, 0, incy*Ym, batchCount, opts.queue );

            const magmaFloatComplex** dA_array = (const magmaFloatComplex**) d_A_array;
            const magmaFloatComplex** dX_array = (const magmaFloatComplex**) d_X_array;
            const magmaFloatComplex* dA = (const magmaFloatComplex*) d_A;
            const magmaFloatComplex* dX = (const magmaFloatComplex*) d_X;

            magma_time = magma_sync_wtime( opts.queue );
            if( opts.version == 1 ) {
                magmablas_cgemv_batched(opts.transA, M, N,
                    alpha, dA_array, ldda,
                           dX_array, incx,
                    beta,  d_Y_array, incy,
                    batchCount, opts.queue);
            }
            else{
                magmablas_cgemv_batched_strided(opts.transA, M, N,
                    alpha, dA, ldda, ldda*N,
                           dX, incx, incx*Xm,
                    beta,  d_Y, incy, incy*Ym,
                    batchCount, opts.queue);
            }
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_cgetvector( Ym*batchCount, d_Y, incy, h_Ymagma, incy, opts.queue );

            /* =====================================================================
               Performs operation using Vendor BLAS
               =================================================================== */
            magma_csetvector( Ym*batchCount, h_Y, incy, d_Y, incy, opts.queue );

            device_time = magma_sync_wtime( opts.queue );
            if(opts.version == 1) {
                #ifdef MAGMA_HAVE_CUDA
                #if CUDA_VERSION >= 11070
                cublasCgemvBatched(opts.handle, cublas_trans_const(opts.transA),
                                      M, N,
                                      (const cuFloatComplex *)&alpha,
                                      (const cuFloatComplex **)d_A_array, ldda,
                                      (const cuFloatComplex **)d_X_array, incx,
                                      (const cuFloatComplex *)&beta,
                                      (cuFloatComplex **)d_Y_array, incy, batchCount);
                #else
                for(magma_int_t s = 0; s < batchCount; s++) {
                    magma_cgemv( opts.transA, M, N,
                         alpha, d_A + s*ldda*N,  ldda,
                                d_X + s*Xm*incx, incx,
                         beta,  d_Y + s*Ym*incy, incy, opts.queue );
                }
                #endif
                #else
                hipblasCgemvBatched(opts.handle, hipblas_trans_const(opts.transA),
                                      M, N,
                                      (const hipblasComplex *)&alpha,
                                      (const hipblasComplex **)d_A_array, ldda,
                                      (const hipblasComplex **)d_X_array, incx,
                                      (const hipblasComplex *)&beta,
                                      (hipblasComplex **)d_Y_array, incy, batchCount);
                #endif
            }
            else{
                #ifdef MAGMA_HAVE_CUDA
                #if CUDA_VERSION >= 11070
                cublasCgemvStridedBatched(opts.handle, cublas_trans_const(opts.transA),
                                      M, N,
                                      (const cuFloatComplex *)&alpha,
                                      (const cuFloatComplex *)d_A, ldda, ldda*N,
                                      (const cuFloatComplex *)d_X, incx, incx*Xm,
                                      (const cuFloatComplex *)&beta,
                                      (cuFloatComplex *)d_Y, incy, incy*Ym, batchCount);
                #else
                for(magma_int_t s = 0; s < batchCount; s++) {
                    magma_cgemv( opts.transA, M, N,
                         alpha, d_A + s*ldda*N,  ldda,
                                d_X + s*Xm*incx, incx,
                         beta,  d_Y + s*Ym*incy, incy, opts.queue );
                }
                #endif
                #else
                hipblasCgemvStridedBatched(opts.handle, hipblas_trans_const(opts.transA),
                                      M, N,
                                      (const hipblasComplex *)&alpha,
                                      (const hipblasComplex *)d_A, ldda, ldda*N,
                                      (const hipblasComplex *)d_X, incx, incx*Xm,
                                      (const hipblasComplex *)&beta,
                                      (hipblasComplex *)d_Y, incy, incy*Ym, batchCount);
                #endif
            }
            device_time = magma_sync_wtime( opts.queue ) - device_time;
            device_perf = gflops / device_time;
            magma_cgetvector( Ym*batchCount, d_Y, incy, h_Ydevice, incy, opts.queue );

            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    blasf77_cgemv( lapack_trans_const(opts.transA),
                                   &M, &N,
                                   &alpha, h_A + i*lda*N, &lda,
                                           h_X + i*Xm*incx, &incx,
                                   &beta,  h_Y + i*Ym*incy, &incy );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = Xn
                magma_error  = 0;
                device_error = 0;

                for (int s=0; s < batchCount; s++){
                    normalize = sqrt(float(Xm+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    blasf77_caxpy( &Ym, &c_neg_one, &h_Y[s*Ym*incy], &incy, &h_Ymagma[s*Ym*incy], &incy );
                    blasf77_caxpy( &Ym, &c_neg_one, &h_Y[s*Ym*incy], &incy, &h_Ydevice[s*Ym*incy], &incy );
                    error = lapackf77_clange( "F", &ione, &Ym, &h_Ymagma[s*Ym*incy], &incy, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                    error = lapackf77_clange( "F", &ione, &Ym, &h_Ydevice[s*Ym*incy], &incy, work )
                          / normalize;
                    device_error = magma_max_nan( error, device_error );
                }

                bool okay = (magma_error < tol && device_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e      %8.2e  %s\n",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf,  1000.*magma_time,
                       device_perf,  1000.*device_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, device_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)       ---   (  ---  )     ---       ---\n",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf,  1000.*magma_time,
                       device_perf,  1000.*device_time);
            }

            magma_free_cpu( h_A );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Y );
            magma_free_cpu( h_Ymagma );
            magma_free_cpu( h_Ydevice );

            magma_free( d_A );
            magma_free( d_X );
            magma_free( d_Y );
            magma_free( d_A_array );
            magma_free( d_X_array );
            magma_free( d_Y_array );

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
