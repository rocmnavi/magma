#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023
       
       @author Azzam Haidar
       @author Tingxing Dong

*/

#include "magma_internal.h"

/******************************************************************************/
static
__global__ void stepinit_ipiv_kernel(magma_int_t **ipiv_array, int pm)
{
    magma_int_t *ipiv = ipiv_array[blockIdx.x];

    int tx = threadIdx.x;  
#if 0
    // best case senario piv = i ==> no piv
    // set piv equal to myself piv[i]=i
    if (tx < pm)
    {
        ipiv[tx] = tx+1;
    }
#else
    //set piv from the last to the first shifted by 32 such a way that it simulate the worst case
    if (tx < pm)
    {
        int i, s;
        i = pm/32;
        i = (i == 1 ? 0 : i);
        s = tx%i;
        ipiv[tx] =  ( (pm - (s*32) ) - tx/i);
        //printf("voici s %d pm %d me %d  ipiv %d \n",s, pm, tx, ipiv[tx]);
    }
#endif
}


/******************************************************************************/
extern "C"
void stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount, magma_queue_t queue)

{
    hipLaunchKernelGGL(stepinit_ipiv_kernel, dim3(batchCount), dim3(pm), 0, queue->hip_stream() , ipiv_array, pm);
}


/******************************************************************************/
static
__global__ void magma_iset_pointer_kernel(
    magma_int_t **output_array,
    magma_int_t *input,
    int lda,
    int row, int column, 
    int batchSize)
{
    output_array[blockIdx.x] =  input + blockIdx.x * batchSize + row + column * lda;
}


/******************************************************************************/
extern "C"
void magma_iset_pointer(
    magma_int_t **output_array,
    magma_int_t *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchSize,
    magma_int_t batchCount, magma_queue_t queue)
{
    /*
    convert consecutive stored variable to array stored
    for example the size  of A is N*batchCount; N is the size of A(batchSize)
    change into A_array[0] A_array[1],... A_array[batchCount-1], where the size of each A_array[i] is N
    */
    hipLaunchKernelGGL(magma_iset_pointer_kernel, dim3(batchCount), dim3(1), 0, queue->hip_stream() , output_array, input, lda,  row, column, batchSize);
}



/******************************************************************************/
__global__ void idisplace_pointers_kernel(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column)
{
    magma_int_t *inpt = input_array[blockIdx.x];
    output_array[blockIdx.x] = &inpt[row + column * lda];
    //printf("==> zdisplace_pointer_kernel input %p input_array %p output_array %p  \n",inpt, input_array[blockIdx.x],output_array[blockIdx.x]);
}


/******************************************************************************/
extern "C"
void magma_idisplace_pointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue)

{
    /*
    compute the offset for all the matrices and save the displacment of the new pointer on output_array.
    input_array contains the pointers to the initial position.
    output_array[i] = input_array[i] + row + lda * column; 
    */
    hipLaunchKernelGGL(idisplace_pointers_kernel, dim3(batchCount), dim3(1), 0, queue->hip_stream() , output_array, input_array, lda, row, column);
}
