#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/ztrtri_lower.hip.cpp, normal z -> s, Thu Jan 25 22:57:57 2024

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       This file implements lower case, and is called by strtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "strtri.hip.hpp"
#include "strtri_lower_device.hip.hpp"


/******************************************************************************/
__global__ void
strtri_diag_lower_kernel(
    magma_diag_t diag, int n, const float *A, int lda, float *d_dinvA)
{
    strtri_diag_lower_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_sgemm16_part1_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm16_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm16_part2_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm16_part2_lower_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm32_part1_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm32_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm32_part2_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm32_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm64_part1_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm64_part2_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part1_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part2_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part3_lower_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part3_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}
