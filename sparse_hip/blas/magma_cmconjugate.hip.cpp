#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from sparse_hip/blas/magma_zmconjugate.hip.cpp, normal z -> c, Thu Jan 25 22:58:37 2024
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


__global__ void 
magma_cmconjugate_kernel(  
    int num_rows,
    magma_index_t *rowptr, 
    magmaFloatComplex *values )
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < num_rows ){
        for( int i = rowptr[row]; i < rowptr[row+1]; i++){
            values[i] = MAGMA_C_CONJ( values[i] );
        }
    }
}



/**
    Purpose
    -------

    This function conjugates a matrix. For a real matrix, no value is changed.

    Arguments
    ---------

    @param[in,out]
    A           magma_c_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmconjugate(
    magma_c_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    dim3 grid( magma_ceildiv( A->num_rows, BLOCK_SIZE ));
    hipLaunchKernelGGL(magma_cmconjugate_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, queue->hip_stream() ,  A->num_rows, A->drow, A->dval );
        
    return info;
}
