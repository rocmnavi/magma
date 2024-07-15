/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from testing/testing_zgbtrf_batched.cpp, normal z -> d, Mon Jul 15 16:58:05 2024
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

// On input, LUB and IPIV is LU factorization of AB.
// Requires m == n.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    magma_int_t M, magma_int_t N,
    magma_int_t KL, magma_int_t KU,
    double *AB,  magma_int_t LDAB,
    double *LUB, magma_int_t *IPIV )
{
    if ( M != N ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }

    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const magma_int_t ione = 1;

    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info = 0;
    double *x, *b;

    // initialize RHS
    TESTING_CHECK( magma_dmalloc_cpu( &x, N ));
    TESTING_CHECK( magma_dmalloc_cpu( &b, N ));
    lapackf77_dlarnv( &ione, ISEED, &N, b );
    blasf77_dcopy( &N, b, &ione, x, &ione );

    // solve Ax = b
    lapackf77_dgbtrs(MagmaNoTransStr, &N, &KL, &KU, &ione, LUB, &LDAB, IPIV, x, &N, &info );
    if (info != 0) {
        printf("lapackf77_dgbtrs returned error %lld: %s.\n",
               (long long) info, magma_strerror( info ));
    }

    // compute r = Ax - b, saved in b
    blasf77_dgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                           &c_one,     AB + KL , &LDAB,
                                       x       , &ione,
                           &c_neg_one, b       , &ione);

    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_dlangb( "F", &N, &KL, &KU, AB + KL, &LDAB, work);
    norm_r = lapackf77_dlange( "F", &N, &ione, b, &N, work );
    norm_x = lapackf77_dlange( "F", &N, &ione, x, &N, work );

    magma_free_cpu( x );
    magma_free_cpu( b );

    return norm_r / (N * norm_A * norm_x);
}


// On input, LUB and IPIV is LU factorization of A.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 4 more matrices, in dense format (not in band format)

double get_band_LU_error(
            magma_int_t M, magma_int_t N,
            magma_int_t KL, magma_int_t KU,
            double *AB,  magma_int_t ldab,
            double *LUB, magma_int_t *IPIV)
{
#define   A(i,j)   A[(j)*M    + (i)]
#define  LU(i,j)  LU[(j)*M    + (i)]
#define  AB(i,j)  AB[(j)*ldab + (i)]
#define LUB(i,j) LUB[(j)*ldab + (i)]

    magma_int_t min_mn = min(M, N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    double alpha = MAGMA_D_ONE;
    double beta  = MAGMA_D_ZERO;
    double *A, *LU, *L, *U;
    double work[1], matnorm, residual;

    TESTING_CHECK( magma_dmalloc_cpu( &A,  M*N ));
    TESTING_CHECK( magma_dmalloc_cpu( &LU, M*N ));
    TESTING_CHECK( magma_dmalloc_cpu( &L,  M*min_mn ));
    TESTING_CHECK( magma_dmalloc_cpu( &U,  min_mn*N ));
    memset( A,  0, M*N*sizeof(double) );
    memset( LU, 0, M*N*sizeof(double) );
    memset( L,  0, M*min_mn*sizeof(double) );
    memset( U,  0, min_mn*N*sizeof(double) );

    // recover A in dense form, account for extra KL super-diagonals
    for(j = 0; j < N; j++) {
        int col_start      = max(0, j-KU);
        int col_end        = min(j+KL,M-1);
        int col_length     = col_end - col_start + 1;
        int col_start_band = KL + max(KU-j,0);
        memcpy( &A(col_start,j), &AB(col_start_band,j), col_length * sizeof(double));
    }
    // end of converting AB to dense in A

    // recover LU in dense form
    magma_int_t KV = KL + KU;
    for(j = 0; j < N; j++) {
        magma_int_t col_start      = max(0, j-KV);
        magma_int_t col_end        = min(j+KL,M-1);
        magma_int_t col_length     = col_end - col_start + 1;
        magma_int_t col_start_band = max(KV-j,0);
        memcpy( &LU(col_start,j), &LUB(col_start_band,j), col_length * sizeof(double));
    }

    // swapping to recover L
    for(j = 0; j < N-2; j++) {
        const magma_int_t k1 = j+2;
        const magma_int_t k2 = N;
        lapackf77_dlaswp(&ione, &LU(0,j), &M, &k1, &k2, IPIV, &ione );
    }
    // end of converting LUB to dense in LU

    lapackf77_dlaswp( &N, A, &M, &ione, &min_mn, IPIV, &ione);
    lapackf77_dlacpy( MagmaLowerStr, &M, &min_mn, LU, &M, L, &M      );
    lapackf77_dlacpy( MagmaUpperStr, &min_mn, &N, LU, &M, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_D_MAKE( 1., 0. );

    matnorm = lapackf77_dlange("f", &M, &N, A, &M, work);

    blasf77_dgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &M);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*M] = MAGMA_D_SUB( LU[i+j*M], A[i+j*M] );
        }
    }
    residual = lapackf77_dlange("f", &M, &N, LU, &M, work);

    magma_free_cpu( A );
    magma_free_cpu( LU );
    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);

#undef A
#undef LU
#undef AB
#undef LUB
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time=0, cpu_perf=0, cpu_time=0;
    double          error;
    double *h_A, *h_R, *h_Amagma;
    double *dA;
    double **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;

    magma_int_t M, N, Mband, Nband, KL, KU, n2, ldab, lddab, min_mn, info = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    batchCount = opts.batchcount;
    KL         = opts.kl;
    KU         = opts.ku;
    magma_int_t columns;

    printf("%% ## INFO ##: Gflop/s calculation is not available\n");
    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   |Ax-b|/(N*|A|*|x|)\n");
    printf("%%=======================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);

            Mband  = KL + 1 + (KL+KU); // need extra KL for the upper factor
            Nband  = N;
            ldab   = Mband;
            n2     = ldab * Nband * batchCount;
            lddab  = magma_roundup( Mband, opts.align );  // multiple of 32 by default
            gflops = 0.;    // TODO: gflop formula for gbtrf?

            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn * batchCount ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_A,  n2     ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_Amagma,  n2     ));
            TESTING_CHECK( magma_dmalloc_pinned( &h_R,  n2     ));

            TESTING_CHECK( magma_dmalloc( &dA,  lddab * Nband * batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv_magma,  min_mn * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(double*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            // random initialization of h_A seems to produce
            // some matrices that are singular, the additive statements below
            // seem to avoid that
            #pragma omp parallel for schedule(dynamic)
            for(int s = 0; s < batchCount; s++) {
                double* hA = h_A + s*ldab*N;
                for(int j = 0; j < ldab*N; j++) {
                    MAGMA_D_REAL( hA[j] ) += 20.;
                    #if defined(PRECISION_c) || defined(PRECISION_z)
                    MAGMA_D_IMAG( hA[j] ) += 20.;
                    #endif
                }
            }

            columns = Nband * batchCount;
            lapackf77_dlacpy( MagmaFullStr, &Mband, &columns, h_A, &ldab, h_R, &ldab );


            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_dsetmatrix( Mband, columns, h_R, ldab, dA, lddab, opts.queue );
            magma_dset_pointer( dA_array, dA, lddab, 0, 0, lddab*Nband, batchCount, opts.queue );
            magma_iset_pointer( dipiv_array, dipiv_magma, 1, 0, 0, min_mn, batchCount, opts.queue );

            if(opts.version == 1) {
                // top-level API accepting ptr arrays
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_dgbtrf_batched(
                        M, N, KL, KU,
                        dA_array, lddab, dipiv_array, dinfo_magma,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            else if(opts.version == 2) {
                // top-level API accepting (ptr+stride)
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_dgbtrf_batched_strided(
                        M, N, KL, KU,
                        dA, lddab, lddab*Nband,
                        dipiv_magma, min_mn,
                        dinfo_magma, batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            else if(opts.version == 3) {
                // async API accepting ptr arrays and workspace
                // query workspace
                magma_int_t lwork[1] = {-1};
                magma_dgbtrf_batched_work(
                    M, N, KL, KU,
                    NULL, lddab, NULL, NULL,
                    NULL, lwork,
                    batchCount, opts.queue);

                void* device_work = NULL;
                magma_malloc((void**)&device_work, lwork[0]);

                // timing async call only
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_dgbtrf_batched_work(
                    M, N, KL, KU,
                    dA_array, lddab, dipiv_array, dinfo_magma,
                    device_work, lwork,
                    batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_free( device_work );
            }
            else if(opts.version == 4) {
                // async API accepting (ptr+stride) and workspace
                // query workspace
                magma_int_t lwork[1] = {-1};
                magma_dgbtrf_batched_strided_work(
                    M, N, KL, KU,
                    NULL, lddab, lddab*Nband,
                    NULL, min_mn,
                    NULL, NULL, lwork,
                    batchCount, opts.queue);

                void* device_work = NULL;
                magma_malloc((void**)&device_work, lwork[0]);

                // timing async call only
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_dgbtrf_batched_strided_work(
                        M, N, KL, KU,
                        dA, lddab, lddab*Nband,
                        dipiv_magma, min_mn,
                        dinfo_magma,
                        device_work, lwork,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_free( device_work );
            }
            else if(opts.version == 5) {
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_dgbtrf_batched_sliding_window_loopin(
                        M,  N, KL, KU,
                        dA_array, lddab, dipiv_array,
                        dinfo_magma, batchCount, opts.queue );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }

            magma_perf = gflops / magma_time;
            magma_dgetmatrix( Mband, Nband*batchCount, dA, lddab, h_Amagma, ldab, opts.queue );

            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );
            if (info != 0) {
                printf("magma_dgbtrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }
            else {
                for (int i=0; i < batchCount; i++) {
                    if (cpu_info[i] != 0 ) {
                        printf("magma_dgbtrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                    }
                }
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t lapack_threads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(lapack_threads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_dgbtrf(&M, &N, &KL, &KU, h_A + s * ldab * Nband, &ldab, ipiv + s * min_mn, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_dgbtrf matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(lapack_threads);
                #endif

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000  );
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf, magma_time*1000. );
            }

            if ( opts.check ) {
                if( info < 0 ) {
                    error = -1;
                }
                else {
                    magma_getvector( min_mn * batchCount, sizeof(magma_int_t), dipiv_magma, 1, ipiv, 1, opts.queue );
                    error = 0;
                    bool pivot_ok = true;
                    #pragma omp parallel for reduction(max:error)
                    for (int i=0; i < batchCount; i++) {
                        double err = 0;
                        for (int k=0; k < min_mn; k++) {
                            if (ipiv[i*min_mn+k] < 1 || ipiv[i*min_mn+k] > M ) {
                                printf("error for matrix %lld ipiv @ %lld = %lld, terminated on first occurrence\n",
                                        (long long) i, (long long) k, (long long) ipiv[i*min_mn+k] );
                                pivot_ok = false;
                                err      = -1;
                                break;
                            }
                        }

                        if(pivot_ok && err == 0) {
                            //err = get_band_LU_error(M, N, KL, KU, h_R + i * ldab*N,  ldab, h_Amagma + i * ldab*N, ipiv + i * min_mn);
                            err = get_residual(M, N, KL, KU, h_R + i * ldab*N,  ldab, h_Amagma + i * ldab*N, ipiv + i * min_mn);
                            if (std::isnan(err) || std::isinf(err)) {
                                error = err;
                            }
                            else {
                                error = magma_max_nan( err, error );
                            }
                        }
                        else {
                            error = -1;
                        }
                    }
                }
                bool okay = ( error >= 0 && error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( cpu_info );
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free_cpu( h_Amagma );
            magma_free_pinned( h_R );

            magma_free( dA );
            magma_free( dinfo_magma );
            magma_free( dipiv_magma );
            magma_free( dipiv_array );
            magma_free( dA_array );
            fflush( stdout );
        }

        if ( opts.niter > 1 ) {
            printf( "\n" );
        }

    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
