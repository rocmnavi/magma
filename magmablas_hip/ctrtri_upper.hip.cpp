#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magmablas_hip/ztrtri_upper.hip.cpp, normal z -> c, Mon Oct 28 11:12:13 2024

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
       This file implements upper case, and is called by ctrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "ctrtri.hip.hpp"
#include "ctrtri_upper_device.hip.hpp"


/******************************************************************************/
__global__ void
ctrtri_diag_upper_kernel(
    magma_diag_t diag, int n, const magmaFloatComplex *A, int lda, magmaFloatComplex *d_dinvA)
{
    ctrtri_diag_upper_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_cgemm16_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm16_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm16_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm16_part2_upper_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm32_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm32_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm32_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm32_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm64_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm64_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part3_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part3_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}
