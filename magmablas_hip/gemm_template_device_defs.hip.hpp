/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Ahmad Abdelfattah
       @author Azzam Haidar

*/

#ifndef GEMM_TEMPLATE_DEVICE_DEFS_H
#define GEMM_TEMPLATE_DEVICE_DEFS_H

// =============================================================================
// conjugation -- double complex
template<const int conjugate>
__device__ inline
magmaDoubleComplex conj(magmaDoubleComplex &x) {return MAGMA_Z_CONJ(x);}

template<>
__device__ inline
magmaDoubleComplex conj<0>(magmaDoubleComplex &x) {return x;}

// conjugation -- single complex
template<const int conjugate>
__device__ inline
magmaFloatComplex conj(magmaFloatComplex &x) {return MAGMA_C_CONJ(x);}

template<>
__device__ inline
magmaFloatComplex conj<0>(magmaFloatComplex &x) {return x;}

// conjugation -- real single & double
template<const int conjugate>
__device__ static inline
double conj(double &x) {return x;}

template<const int conjugate>
__device__ static inline
float conj(float &x) {return x;}


// =============================================================================
#define fetch(A, m, n, bound)  offs_d##A[min(n*LD##A+m, bound)]

// =============================================================================
#if defined(PRECISION_z)
    #define add(A, B)        MAGMA_Z_ADD(A, B)
    #define mul(A, B)        MAGMA_Z_MUL(A, B)
    #define div(A, B)        MAGMA_Z_DIV(A, B)
    #define fma(A, B, C) C = magmaCfma(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_Z_MAKE(x, y)
#elif defined(PRECISION_c)
    #define add(A, B)        MAGMA_C_ADD(A, B)
    #define mul(A, B)        MAGMA_C_MUL(A, B)
    #define div(A, B)        MAGMA_C_DIV(A, B)
    #define fma(A, B, C) C = magmaCfmaf(A, B, C)
    #define make_FloatingPoint(x, y) MAGMA_C_MAKE(x, y)
#elif defined(PRECISION_h)
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) ((magmaHalf)x)
#else
    #define add(A, B)         (A+B)
    #define mul(A, B)         (A*B)
    #define div(A, B)         (A/B)
    #define fma(A, B, C) C += (A*B)
    #define make_FloatingPoint(x, y) (x)
#endif

#if defined(PRECISION_z)
    #define magmablas_atomic_add magmablas_zatomic_add
#elif defined(PRECISION_c)
    #define magmablas_atomic_add magmablas_catomic_add
#elif defined(PRECISION_d)
    #define magmablas_atomic_add magmablas_datomic_add
#else
    #define magmablas_atomic_add magmablas_satomic_add
#endif

#endif // GEMM_TEMPLATE_DEVICE_DEFS_H
