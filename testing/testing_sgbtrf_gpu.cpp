/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from testing/testing_zgbtrf_gpu.cpp, normal z -> s, Mon Jul 15 16:57:58 2024
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
float get_residual(
    magma_int_t M, magma_int_t N,
    magma_int_t KL, magma_int_t KU,
    float *AB,  magma_int_t LDAB,
    float *LUB, magma_int_t *IPIV )
{
    if ( M != N ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }

    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;

    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info = 0;
    float *x, *b;

    // initialize RHS
    TESTING_CHECK( magma_smalloc_cpu( &x, N ));
    TESTING_CHECK( magma_smalloc_cpu( &b, N ));
    lapackf77_slarnv( &ione, ISEED, &N, b );
    blasf77_scopy( &N, b, &ione, x, &ione );

    // solve Ax = b
    lapackf77_sgbtrs(MagmaNoTransStr, &N, &KL, &KU, &ione, LUB, &LDAB, IPIV, x, &N, &info );
    if (info != 0) {
        printf("lapackf77_sgbtrs returned error %lld: %s.\n",
               (long long) info, magma_strerror( info ));
    }

    // compute r = Ax - b, saved in b
    blasf77_sgbmv( MagmaNoTransStr, &N, &N, &KL, &KU,
                           &c_one,     AB + KL , &LDAB,
                                       x       , &ione,
                           &c_neg_one, b       , &ione);

    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_slangb( "F", &N, &KL, &KU, AB + KL, &LDAB, work);
    norm_r = lapackf77_slange( "F", &N, &ione, b, &N, work );
    norm_x = lapackf77_slange( "F", &N, &ione, x, &N, work );

    magma_free_cpu( x );
    magma_free_cpu( b );

    return norm_r / (N * norm_A * norm_x);
}


// On input, LUB and IPIV is LU factorization of A.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 4 more matrices, in dense format (not in band format)
float get_band_LU_error(
            magma_int_t M, magma_int_t N,
            magma_int_t KL, magma_int_t KU,
            float *AB,  magma_int_t ldab,
            float *LUB, magma_int_t *IPIV)
{
#define   A(i,j)   A[(j)*M    + (i)]
#define  LU(i,j)  LU[(j)*M    + (i)]
#define  AB(i,j)  AB[(j)*ldab + (i)]
#define LUB(i,j) LUB[(j)*ldab + (i)]

    magma_int_t min_mn = min(M, N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    float alpha = MAGMA_S_ONE;
    float beta  = MAGMA_S_ZERO;
    float *A, *LU, *L, *U;
    float work[1], matnorm, residual;

    TESTING_CHECK( magma_smalloc_cpu( &A,  M*N ));
    TESTING_CHECK( magma_smalloc_cpu( &LU, M*N ));
    TESTING_CHECK( magma_smalloc_cpu( &L,  M*min_mn ));
    TESTING_CHECK( magma_smalloc_cpu( &U,  min_mn*N ));
    memset( A,  0, M*N*sizeof(float) );
    memset( LU, 0, M*N*sizeof(float) );
    memset( L,  0, M*min_mn*sizeof(float) );
    memset( U,  0, min_mn*N*sizeof(float) );

    // recover A in dense form, account for extra KL super-diagonals
    #pragma omp parallel for
    for(j = 0; j < N; j++) {
        int col_start      = max(0, j-KU);
        int col_end        = min(j+KL,M-1);
        int col_length     = col_end - col_start + 1;
        int col_start_band = KL + max(KU-j,0);
        memcpy( &A(col_start,j), &AB(col_start_band,j), col_length * sizeof(float));
    }
    // end of converting AB to dense in A

    // recover LU in dense form
    magma_int_t KV = KL + KU;
    #pragma omp parallel for
    for(j = 0; j < N; j++) {
        magma_int_t col_start      = max(0, j-KV);
        magma_int_t col_end        = min(j+KL,M-1);
        magma_int_t col_length     = col_end - col_start + 1;
        magma_int_t col_start_band = max(KV-j,0);
        memcpy( &LU(col_start,j), &LUB(col_start_band,j), col_length * sizeof(float));
    }

    // swapping to recover L
    for(j = 0; j < N-2; j++) {
        const magma_int_t k1 = j+2;
        const magma_int_t k2 = N;
        lapackf77_slaswp(&ione, &LU(0,j), &M, &k1, &k2, IPIV, &ione );
    }
    // end of converting LUB to dense in LU

    lapackf77_slaswp( &N, A, &M, &ione, &min_mn, IPIV, &ione);
    lapackf77_slacpy( MagmaLowerStr, &M, &min_mn, LU, &M, L, &M      );
    lapackf77_slacpy( MagmaUpperStr, &min_mn, &N, LU, &M, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_S_MAKE( 1., 0. );

    matnorm = lapackf77_slange("f", &M, &N, A, &M, work);

    blasf77_sgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &M);

    #pragma omp parallel for
    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*M] = MAGMA_S_SUB( LU[i+j*M], A[i+j*M] );
        }
    }
    residual = lapackf77_slange("f", &M, &N, LU, &M, work);

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
   -- Testing sgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time=0, cpu_perf=0, cpu_time=0;
    float          error;
    float *h_A, *h_R, *h_Amagma;
    float *dA;

    magma_int_t     *ipiv;
    magma_int_t     *dipiv_magma, *dinfo_magma;

    magma_int_t M, N, Mband, Nband, KL, KU, n2, ldab, lddab, min_mn, info = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    float tol = opts.tolerance * lapackf77_slamch("E");

    KL = opts.kl;
    KU = opts.ku;

    printf("%% ## INFO ##: Gflop/s calculation is not available\n");
    printf("%% Lower bandwidth (KL) = %lld\n", (long long)KL);
    printf("%% Upper bandwidth (KU) = %lld\n", (long long)KU);
    printf("%% M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   |Ax-b|/(N*|A|*|x|)\n");
    printf("%%=======================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);

            Mband  = KL + 1 + (KL+KU); // need extra KL for the upper factor
            Nband  = N;
            ldab   = Mband;
            n2     = ldab * Nband;
            lddab  = magma_roundup( Mband, opts.align );  // multiple of 32 by default
            gflops = 0.;    // TODO: gflop formula for gbtrf?

            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn ));
            TESTING_CHECK( magma_smalloc_cpu( &h_A,  n2 ));
            TESTING_CHECK( magma_smalloc_cpu( &h_Amagma, n2 ));
            TESTING_CHECK( magma_smalloc_cpu( &h_R, n2 ));

            TESTING_CHECK( magma_smalloc( &dA,  lddab * Nband ));
            TESTING_CHECK( magma_imalloc( &dipiv_magma,  min_mn ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  1 ));

            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            // random initialization of h_A seems to produce
            // some matrices that are singular, the additive statements below
            // seem to avoid that
            #pragma omp parallel for schedule(dynamic)
            for(int j = 0; j < ldab*N; j++) {
                MAGMA_S_REAL( h_A[j] ) += 20.;
                #if defined(PRECISION_c) || defined(PRECISION_z)
                MAGMA_S_IMAG( h_A[j] ) += 20.;
                #endif
            }

            lapackf77_slacpy( MagmaFullStr, &Mband, &Nband, h_A, &ldab, h_R, &ldab );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_ssetmatrix( Mband, Nband, h_R, ldab, dA, lddab, opts.queue );
            if(opts.version == 1) {
                // sync. interface
                magma_time = magma_wtime();
                magma_sgbtrf_native(
                    M, N, KL, KU,
                    dA, lddab, dipiv_magma, &info);
                magma_time = magma_wtime() - magma_time;
            }
            else if (opts.version == 2){
                // async. interface
                // query workspace first
                magma_int_t lwork[1] = {-1};
                magma_sgbtrf_native_work(
                    M, N, KL, KU,
                    NULL, lddab,
                    NULL, &info,
                    NULL, lwork, opts.queue);

                void* device_work = NULL;
                TESTING_CHECK( magma_malloc(&device_work, lwork[0]) );

                // time the async call only
                magma_time = magma_sync_wtime( opts.queue );
                magma_sgbtrf_native_work(
                    M, N, KL, KU,
                    dA, lddab, dipiv_magma, &info,
                    device_work, lwork, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            else if(opts.version == 3) {
                magma_time = magma_wtime();
                magma_sgbtf2_native_v2(M, N, KL, KU, dA, lddab, dipiv_magma, &info, opts.queue);
                magma_time = magma_wtime() - magma_time;
            }
            magma_perf = gflops / magma_time;
            magma_sgetmatrix( Mband, Nband, dA, lddab, h_Amagma, ldab, opts.queue );

            if (info != 0) {
                printf("magma_sgbtrf_gpu returned internal error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sgbtrf(&M, &N, &KL, &KU, h_A, &ldab, ipiv, &info );
                if (info != 0) {
                    printf("lapackf77_sgbtrf returned error %lld: %s.\n", (long long)info, magma_strerror( info ));
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)",
                       (long long) M, (long long) N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000  );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)",
                       (long long) M, (long long) N,
                       magma_perf, magma_time*1000. );
            }

            if ( opts.check ) {
                if( info != 0 ) {
                    error = -1;
                }
                else {
                    magma_getvector( min_mn, sizeof(magma_int_t), dipiv_magma, 1, ipiv, 1, opts.queue );
                    error = 0;
                    bool pivot_ok = true;
                    for (int k=0; k < min_mn; k++) {
                        if (ipiv[k] < 1 || ipiv[k] > M ) {
                            printf("error: ipiv @ %lld = %lld, terminated on first occurrence\n",
                                    (long long) k, (long long) ipiv[k] );
                            pivot_ok = false;
                            error      = -1;
                            break;
                        }
                    }

                    if(pivot_ok && error == 0) {
                        if (M == N) {
                            error = get_residual(M, N, KL, KU, h_R,  ldab, h_Amagma, ipiv );
                        }
                        else {
                            printf("  [INFO]: residual check defined only for square matrices ");
                            error = 0;
                        }
                    }
                    else {
                        error = -1;
                    }
                }
                bool okay = ( error >= 0 && error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free_cpu( h_Amagma );
            magma_free_cpu( h_R );

            magma_free( dA );
            magma_free( dinfo_magma );
            magma_free( dipiv_magma );
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
