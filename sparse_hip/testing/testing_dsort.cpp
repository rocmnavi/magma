/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from sparse_hip/testing/testing_zsort.cpp, normal z -> d, Mon Jul 15 16:58:27 2024
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t i, n=100;
    magma_index_t *x=NULL;
    double *y=NULL;
    
    magma_d_matrix A={Magma_CSR};

    TESTING_CHECK( magma_index_malloc_cpu( &x, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        int r = rand()%100;
        x[i] = r;
        printf("%d  ", x[i]);
    }
    printf("\n\n");
    
    printf("sorting...");
    TESTING_CHECK( magma_dindexsort(x, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%d  ", x[i]);
    }
    printf("\n\n");

    magma_free_cpu( x );
    
    
    TESTING_CHECK( magma_dmalloc_cpu( &y, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        double r = (double) rand()/(double) 10.;
        y[i] = MAGMA_D_MAKE( r, 0.0);
        if (i % 5 == 0)
            y[i] = - y[i];
        printf("%2.2f + %2.2f  ", MAGMA_D_REAL(y[i]), MAGMA_D_IMAG(y[i]) );
    }
    printf("\n\n");
    
    printf("sorting...");
    TESTING_CHECK( magma_dsort(y, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%2.2f + %2.2f  ", MAGMA_D_REAL(y[i]), MAGMA_D_IMAG(y[i]) );
    }
    printf("\n\n");

    magma_free_cpu( y );
    
    i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_dm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_d_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz);
    
        TESTING_CHECK( magma_index_malloc_cpu( &x, A.num_rows*10 ));
        magma_int_t num_ind = 0;

        TESTING_CHECK( magma_ddomainoverlap( A.num_rows, &num_ind, A.row, A.col, x, queue ));
                printf("domain overlap indices:\n");
        for(magma_int_t j = 0; j < num_ind; j++ ){
            printf("%d  ", x[j] );
        }
        printf("\n\n");
        magma_free_cpu( x );
        magma_dmfree(&A, queue);
        
        i++;
    }

    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
