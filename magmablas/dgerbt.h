/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas/zgerbt.h, normal z -> d, Fri Aug 25 13:19:37 2023

       @author Adrien Remy
       @author Azzam Haidar
       
       Definitions used in dgerbt.cu dgerbt_batched.cu
*/

#ifndef DGERBT_H
#define DGERBT_H

// =============================================================================
// classical prototypes

__global__ void 
magmablas_delementary_multiplication_kernel(
    magma_int_t n,
    double *dA, magma_int_t offsetA, magma_int_t ldda, 
    double *du, magma_int_t offsetu, 
    double *dv, magma_int_t offsetv);

__global__ void 
magmablas_dapply_vector_kernel(
    magma_int_t n,
    double *du, magma_int_t offsetu,  double *db, magma_int_t offsetb );

__global__ void 
magmablas_dapply_transpose_vector_kernel(
    magma_int_t n,
    double *du, magma_int_t offsetu, double *db, magma_int_t offsetb );

// =============================================================================
// batched prototypes

__global__ void 
magmablas_delementary_multiplication_kernel_batched(
    magma_int_t n,
    double **dA_array, magma_int_t offsetA, magma_int_t ldda, 
    double *du, magma_int_t offsetu, 
    double *dv, magma_int_t offsetv);

__global__ void 
magmablas_dapply_vector_kernel_batched(
    magma_int_t n,
    double *du, magma_int_t offsetu, double **db_array, magma_int_t offsetb );

__global__ void 
magmablas_dapply_transpose_vector_kernel_batched(
    magma_int_t n,
    double *du, magma_int_t offsetu, double **db_array, magma_int_t offsetb );

#endif // DGERBT_H
