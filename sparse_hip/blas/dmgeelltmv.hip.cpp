#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from sparse_hip/blas/zmgeelltmv.hip.cpp, normal z -> d, Mon Jul 15 16:58:10 2024

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

template<bool betazero>
__global__ void 
dmgeelltmv_kernel( 
        int num_rows, 
        int num_cols,
        int num_vecs,
        int num_cols_per_row,
        double alpha, 
        double * dval, 
        magma_index_t * dcolind,
        double * dx,
        double beta, 
        double * dy)
{
    HIP_DYNAMIC_SHARED( double, dot)
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < num_rows ) {
        for( int i=0; i<num_vecs; i++ )
            dot[ threadIdx.x+ i*blockDim.x ] = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            double val = dval [ num_rows * n + row ];
                for( int i=0; i<num_vecs; i++ )
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                        val * dx[col + i * num_cols ];
        }
        for( int i=0; i<num_vecs; i++ ) {
            if (betazero) {
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] *alpha;
            } else {
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                        * alpha + beta * dy [ row + i*num_cols ];
            }
        }
    }
}


/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELL. 
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 
                
    @param[in]
    num_vecs    mama_int_t
                number of vectors
                
    @param[in]
    nnz_per_row magma_int_t
                number of elements in the longest row 
                
    @param[in]
    alpha       double
                scalar multiplier

    @param[in]
    dval        magmaDouble_ptr
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

    @param[in]
    dx          magmaDouble_ptr
                input vector x

    @param[in]
    beta        double
                scalar multiplier

    @param[out]
    dy          magmaDouble_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                * sizeof( double ); // num_vecs vectors 
    if (beta == MAGMA_D_ZERO) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dmgeelltmv_kernel<true>), dim3(grid), dim3(threads), MEM_SIZE, queue->hip_stream() ,  m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(dmgeelltmv_kernel<false>), dim3(grid), dim3(threads), MEM_SIZE, queue->hip_stream() ,  m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    }


    return MAGMA_SUCCESS;
}
