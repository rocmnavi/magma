#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from sparse_hip/blas/zparic_kernels.hip.cpp, normal z -> d, Mon Jul 15 16:58:12 2024

*/
#include "magmasparse_internal.h"

#define PRECISION_d


__global__ void 
magma_dparic_csr_kernel(    
    magma_int_t n, 
    magma_int_t nnz, 
    magma_index_t *Arowidx, 
    magma_index_t *Acolidx, 
    const double * __restrict__  A_val,
    magma_index_t *rowptr, 
    magma_index_t *colidx, 
    double *val )
{
    int i, j;
    int k = (blockDim.x * blockIdx.x + threadIdx.x); // % nnz;
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    double s, sp;
    int il, iu, jl, ju;
    if ( k < nnz ) {     
        i = Arowidx[k];
        j = Acolidx[k];
#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        s = __ldg( A_val+k );
#else
        s = A_val[k];
#endif
        il = rowptr[i];
        iu = rowptr[j];
        while (il < rowptr[i+1] && iu < rowptr[j+1]) {
            sp = zero;
            jl = colidx[il];
            ju = colidx[iu];
            if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else {
                // we are going to modify this u entry
                sp = val[il] * val[iu];
                s -= sp;
                il++;
                iu++;
            }
        }
        s += sp; // undo the last operation (it must be the last)
        // modify entry
        if (i == j) // diagonal
            val[il-1] = MAGMA_D_MAKE( sqrt( fabs( MAGMA_D_REAL(s) )), 0.0 );
        else  //sub-diagonal
            val[il-1] =  s / val[iu-1];
    }
}// kernel 


/**
    Purpose
    -------
    
    This routine iteratively computes an incomplete LU factorization.
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    This routine was used in the ISC 2015 paper:
    E. Chow et al.: "Asynchronous Iterative Algorithm for Computing Incomplete
                     Factorizations on GPUs", 
                     ISC High Performance 2015, LNCS 9137, pp. 1-16, 2015.
                     
    The input format of the initial guess matrix A is Magma_CSRCOO,
    A_CSR is CSR or CSRCOO format. 

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A - initial guess (lower triangular)

    @param[in,out]
    A_CSR       magma_d_matrix
                input/output matrix containing the IC approximation
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dparic_csr( 
    magma_d_matrix A,
    magma_d_matrix A_CSR,
    magma_queue_t queue )
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv( A.nnz, blocksize1 );
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    
    
    hipLaunchKernelGGL(magma_dparic_csr_kernel, dim3(grid), dim3(block), 0, queue->hip_stream() ,  A.num_rows, A.nnz, 
              A.rowidx, A.col, A.val, 
              A_CSR.row, A_CSR.col,  A_CSR.val );

    return MAGMA_SUCCESS;
}
