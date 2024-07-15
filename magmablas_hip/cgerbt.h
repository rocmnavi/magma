#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/zgerbt.h, normal z -> c, Mon Jul 15 16:58:30 2024

       @author Adrien Remy
       @author Azzam Haidar

       Definitions used in cgerbt.cu cgerbt_batched.cu
*/

#ifndef CGERBT_H
#define CGERBT_H

// =============================================================================
// classical prototypes

__global__ void
magmablas_celementary_multiplication_kernel(
    int Am, int An,
    magmaFloatComplex *dA, int Ai, int Aj, int ldda,
    magmaFloatComplex *du, int Ui,
    magmaFloatComplex *dv, int Vi);

__global__ void
magmablas_capply_vector_kernel(
    int n, int nrhs,
    magmaFloatComplex *du, int offsetu,  magmaFloatComplex *db, int lddb, int offsetb );

__global__ void
magmablas_capply_transpose_vector_kernel(
    int n, int rhs,
    magmaFloatComplex *du, int offsetu, magmaFloatComplex *db, int lddb, int offsetb );

// =============================================================================
// batched prototypes

__global__ void
magmablas_celementary_multiplication_kernel_batched(
    int Am, int An,
    magmaFloatComplex **dA_array, int Ai, int Aj, int ldda,
    magmaFloatComplex *du, int Ui,
    magmaFloatComplex *dv, int Vi);

__global__ void
magmablas_capply_vector_kernel_batched(
    int n, int nrhs,
    magmaFloatComplex *du, int offsetu, magmaFloatComplex **db_array, int lddb, int offsetb );

__global__ void
magmablas_capply_transpose_vector_kernel_batched(
    int n, int nrhs,
    magmaFloatComplex *du, int offsetu, magmaFloatComplex **db_array, int lddb, int offsetb );

#endif // CGERBT_H
