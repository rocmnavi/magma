#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zgerbt.h, normal z -> s, Mon Oct 28 11:13:24 2024

       @author Adrien Remy
       @author Azzam Haidar

       Definitions used in sgerbt.cu sgerbt_batched.cu
*/

#ifndef SGERBT_H
#define SGERBT_H

// =============================================================================
// classical prototypes

__global__ void
magmablas_selementary_multiplication_kernel(
    int Am, int An,
    float *dA, int Ai, int Aj, int ldda,
    float *du, int Ui,
    float *dv, int Vi);

__global__ void
magmablas_sapply_vector_kernel(
    int n, int nrhs,
    float *du, int offsetu,  float *db, int lddb, int offsetb );

__global__ void
magmablas_sapply_transpose_vector_kernel(
    int n, int rhs,
    float *du, int offsetu, float *db, int lddb, int offsetb );

// =============================================================================
// batched prototypes

__global__ void
magmablas_selementary_multiplication_kernel_batched(
    int Am, int An,
    float **dA_array, int Ai, int Aj, int ldda,
    float *du, int Ui,
    float *dv, int Vi);

__global__ void
magmablas_sapply_vector_kernel_batched(
    int n, int nrhs,
    float *du, int offsetu, float **db_array, int lddb, int offsetb );

__global__ void
magmablas_sapply_transpose_vector_kernel_batched(
    int n, int nrhs,
    float *du, int offsetu, float **db_array, int lddb, int offsetb );

#endif // SGERBT_H
