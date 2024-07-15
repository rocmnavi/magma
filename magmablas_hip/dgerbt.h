#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zgerbt.h, normal z -> d, Mon Jul 15 16:58:30 2024

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
    int Am, int An,
    double *dA, int Ai, int Aj, int ldda,
    double *du, int Ui,
    double *dv, int Vi);

__global__ void
magmablas_dapply_vector_kernel(
    int n, int nrhs,
    double *du, int offsetu,  double *db, int lddb, int offsetb );

__global__ void
magmablas_dapply_transpose_vector_kernel(
    int n, int rhs,
    double *du, int offsetu, double *db, int lddb, int offsetb );

// =============================================================================
// batched prototypes

__global__ void
magmablas_delementary_multiplication_kernel_batched(
    int Am, int An,
    double **dA_array, int Ai, int Aj, int ldda,
    double *du, int Ui,
    double *dv, int Vi);

__global__ void
magmablas_dapply_vector_kernel_batched(
    int n, int nrhs,
    double *du, int offsetu, double **db_array, int lddb, int offsetb );

__global__ void
magmablas_dapply_transpose_vector_kernel_batched(
    int n, int nrhs,
    double *du, int offsetu, double **db_array, int lddb, int offsetb );

#endif // DGERBT_H
