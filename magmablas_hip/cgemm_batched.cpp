/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas_hip/zgemm_batched.cpp, normal z -> c, Fri Aug 25 13:17:06 2023

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar

*/
#include "magma_internal.h"
#include "commonblas_c.h"

#define PRECISION_c


/* on some platforms (i.e. hipMAGMA on ROCm stack), we define custom types
 *  * So, to keep the C++ compiler from giving errors, we cast arguments to internal
 *   * BLAS routines. The hipify script should replace `cu*Complex` with appropriate HIP types
 *    *
 *     * FUTURE READERS: If hipBLAS changes numbers to `hipblas*Complex` rather than `hip*Complex`,
 *      *   these will need more complicated macro if/else blocks
 *       */
#ifdef PRECISION_z
  #ifdef MAGMA_HAVE_HIP
    typedef hipblasComplex BackendFloat_t;
  #else
    typedef hipblasComplex BackendFloat_t;
  #endif
#elif defined(PRECISION_c)
  #ifdef MAGMA_HAVE_HIP
    typedef hipblasComplex BackendFloat_t;
  #else
    typedef hipblasFloatComplex BackendFloat_t;
  #endif
#elif defined(PRECISION_d)
  typedef float BackendFloat_t;
#else
  typedef float BackendFloat_t;
#endif



void
magma_cgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t use_cublas  = magma_crecommend_cublas_gemm_batched(transA, transB, m, n, k);
    magma_int_t zero_offset = (Ai == 0 && Aj == 0 && Bi == 0 && Bj == 0 && Ci == 0 && Cj == 0);
    if(use_cublas){
        if(zero_offset){
            hipblasCgemmBatched(
                    queue->hipblas_handle(), hipblas_trans_const(transA), hipblas_trans_const(transB),
                    int(m), int(n), int(k),
                    (BackendFloat_t*)&alpha, (const BackendFloat_t**)dA_array, int(ldda),
                            (const BackendFloat_t**)dB_array, int(lddb),
                    (BackendFloat_t*)&beta,  (BackendFloat_t**)dC_array, int(lddc), int(batchCount) );
        }
        else{
            magmaFloatComplex** dAarray = (magmaFloatComplex**)queue->get_dAarray();
            magmaFloatComplex** dBarray = (magmaFloatComplex**)queue->get_dBarray();
            magmaFloatComplex** dCarray = (magmaFloatComplex**)queue->get_dCarray();
            magma_int_t max_batchCount   = queue->get_maxBatch();
            for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
                magma_int_t batch = min(max_batchCount, batchCount-i);
                magma_cdisplace_pointers(dAarray, (magmaFloatComplex**)dA_array + i, ldda, Ai, Aj, batch, queue);
                magma_cdisplace_pointers(dBarray, (magmaFloatComplex**)dB_array + i, lddb, Bi, Bj, batch, queue);
                magma_cdisplace_pointers(dCarray, (magmaFloatComplex**)dC_array + i, lddc, Ci, Cj, batch, queue);
                hipblasCgemmBatched(
                        queue->hipblas_handle(), hipblas_trans_const(transA), hipblas_trans_const(transB),
                        int(m), int(n), int(k),
                        (BackendFloat_t*)&alpha, (const BackendFloat_t**)dAarray, int(ldda),
                                (const BackendFloat_t**)dBarray, int(lddb),
                        (BackendFloat_t*)&beta,  (BackendFloat_t**)dCarray, int(lddc), int(batch) );
            }
        }
    }
    else{
        magmablas_cgemm_batched_core(
            transA, transB,
            m, n, k,
            alpha, dA_array, Ai, Aj, ldda,
                   dB_array, Bi, Bj, lddb,
            beta,  dC_array, Ci, Cj, lddc,
            batchCount, queue);
    }
}

/***************************************************************************//**
    Purpose
    -------
    CGEMM performs one of the matrix-matrix operations

        C = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.

    Parameters
    ----------
    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.

    @param[in]
    transB  magma_trans_t.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.

    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( A )  and of the  matrix C.  M  must  be at least  zero.

    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( B ) and the number of columns of the matrix C. N must be
            at least zero.

    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( A ) and the number of rows of the matrix op( B ). K must
            be at least  zero.

    @param[in]
    alpha   COMPLEX
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX array A of DIMENSION ( ldda, ka ), where ka is
             k  when  transA = MagmaNoTrans,  and is  m  otherwise.
             Before entry with  transA = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.

    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as declared
            in the calling (sub) program. When  transA = MagmaNoTrans then
            ldda must be at least  max( 1, m ), otherwise  ldda must be at
            least  max( 1, k ).

    @param[in]
    dB_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX array B of DIMENSION ( lddb, kb ), where kb is
             n  when  transB = MagmaNoTrans,  and is  k  otherwise.
             Before entry with  transB = MagmaNoTrans,  the leading  k by n
             part of the array B must contain the matrix B, otherwise
             the leading  n by k  part of the array B must contain  the
             matrix B.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of each array B as declared
            in the calling (sub) program. When  transB = MagmaNoTrans then
            lddb must be at least  max( 1, k ), otherwise  lddb must be at
            least  max( 1, n ).

    @param[in]
    beta    COMPLEX.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX array C of DIMENSION ( lddc, n ).
             Before entry, the leading  m by n  part of the array  C must
             contain the matrix  C,  except when  beta  is zero, in which
             case C need not be set on entry.
             On exit, the array  C  is overwritten by the  m by n  matrix
             ( alpha*op( A )*op( B ) + beta*C ).

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemm_batched
*******************************************************************************/
extern "C" void
magmablas_cgemm_batched( magma_trans_t transA, magma_trans_t transB,
                     magma_int_t m, magma_int_t n, magma_int_t k,
                     magmaFloatComplex alpha,
                     magmaFloatComplex const * const * dA_array, magma_int_t ldda,
                     magmaFloatComplex const * const * dB_array, magma_int_t lddb,
                     magmaFloatComplex beta,
                     magmaFloatComplex **dC_array, magma_int_t lddc,
                     magma_int_t batchCount, magma_queue_t queue )
{
    magmablas_cgemm_batched_core(
                transA, transB, m, n, k,
                alpha, dA_array, 0, 0, ldda,
                       dB_array, 0, 0, lddb,
                 beta, dC_array, 0, 0, lddc,
                batchCount, queue );
}


/******************************************************************************/
extern "C" void
magmablas_cgemm_batched_strided( magma_trans_t transA, magma_trans_t transB,
                     magma_int_t m, magma_int_t n, magma_int_t k,
                     magmaFloatComplex alpha,
                     magmaFloatComplex const * dA, magma_int_t ldda, magma_int_t strideA,
                     magmaFloatComplex const * dB, magma_int_t lddb, magma_int_t strideB,
                     magmaFloatComplex beta,
                     magmaFloatComplex       * dC, magma_int_t lddc, magma_int_t strideC,
                     magma_int_t batchCount, magma_queue_t queue )
{
    magmaFloatComplex** dAarray = (magmaFloatComplex**)queue->get_dAarray();
    magmaFloatComplex** dBarray = (magmaFloatComplex**)queue->get_dBarray();
    magmaFloatComplex** dCarray = (magmaFloatComplex**)queue->get_dCarray();
    magma_int_t max_batchCount   = queue->get_maxBatch();
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t batch = min(max_batchCount, batchCount-i);
        magma_cset_pointer(dAarray, (magmaFloatComplex*)(dA + i * strideA), ldda, 0, 0, strideA, batch, queue);
        magma_cset_pointer(dBarray, (magmaFloatComplex*)(dB + i * strideB), lddb, 0, 0, strideB, batch, queue);
        magma_cset_pointer(dCarray, dC + i * strideC, lddc, 0, 0, strideC, batch, queue);
        magmablas_cgemm_batched_core(
            transA, transB,
            m, n, k,
            alpha, dAarray, 0, 0, ldda,
                   dBarray, 0, 0, lddb,
            beta,  dCarray, 0, 0, lddc,
            batch, queue);
    }
}


/******************************************************************************/
extern "C" void
magma_cgemm_batched( magma_trans_t transA, magma_trans_t transB,
                     magma_int_t m, magma_int_t n, magma_int_t k,
                     magmaFloatComplex alpha,
                     magmaFloatComplex const * const * dA_array, magma_int_t ldda,
                     magmaFloatComplex const * const * dB_array, magma_int_t lddb,
                     magmaFloatComplex beta,
                     magmaFloatComplex **dC_array, magma_int_t lddc,
                     magma_int_t batchCount, magma_queue_t queue )
{
    magma_cgemm_batched_core(
            transA, transB, m, n, k,
            alpha, dA_array, 0, 0, ldda,
                   dB_array, 0, 0, lddb,
            beta,  dC_array, 0, 0, lddc,
            batchCount, queue );
}
