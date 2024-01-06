#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from magmablas_hip/ztranspose_inplace.hip.cpp, normal z -> s, Fri Aug 25 13:17:01 2023

       @author Stan Tomov
       @author Mark Gates
*/
#include "magma_internal.h"

#define NB 16


/******************************************************************************/
// grid is (n/nb) x ((n/nb)/2 + 1), where n/nb is odd.
// lower indicates blocks in lower triangle of grid, including diagonal.
// lower blocks cover left side of matrix, including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width (or width-1)
// to cover right side of matrix.
//      [ A00 A01 A02 ]                  [ A00  .   .  |  .   .  ]
//      [ A10 A11 A12 ]                  [ A10 A11  .  |  .   .  ]
// grid [ A20 A21 A22 ] covers matrix as [ A20 A21 A22 |  .   .  ]
//      [ A30 A31 A32 ]                  [ A30 A31 A32 | A01  .  ]
//      [ A40 A41 A42 ]                  [ A40 A41 A42 | A02 A12 ]
// 
// See stranspose_inplace_even for description of threads.

__global__ void stranspose_inplace_odd(
    int n,
    float *matrix, int lda )
{
    __shared__ float sA[ NB ][ NB+1 ];
    __shared__ float sB[ NB ][ NB+1 ];

    int i = threadIdx.x;
    int j = threadIdx.y;

    bool lower = (blockIdx.x >= blockIdx.y);
    int ii = (lower ? blockIdx.x : (blockIdx.y + gridDim.y - 1));
    int jj = (lower ? blockIdx.y : (blockIdx.x + gridDim.y    ));

    ii *= NB;
    jj *= NB;

    float *A = matrix + ii+i + (jj+j)*lda;
    if ( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        __syncthreads();
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        float *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = *B;
        }
        __syncthreads();
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}


/******************************************************************************/
// grid is ((n/nb) + 1) x (n/nb)/2, where n/nb is even.
// lower indicates blocks in strictly lower triangle of grid, excluding diagonal.
// lower blocks shift up by one to cover left side of matrix including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width
// to cover right side of matrix.
//      [ A00  A01 ]                  [ A10  .  |  .   .  ]
//      [ A10  A11 ]                  [ A20 A21 |  .   .  ]
// grid [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
//      [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
//      [ A40  A41 ]
//
// Each block is NB x NB threads.
// For non-diagonal block A, block B is symmetric block.
// Thread (i,j) loads A(i,j) into sA(j,i) and B(i,j) into sB(j,i), i.e., transposed,
// syncs, then saves sA(i,j) to B(i,j) and sB(i,j) to A(i,j).
// Threads outside the matrix do not touch memory.

__global__ void stranspose_inplace_even(
    int n,
    float *matrix, int lda )
{
    __shared__ float sA[ NB ][ NB+1 ];
    __shared__ float sB[ NB ][ NB+1 ];

    int i = threadIdx.x;
    int j = threadIdx.y;

    bool lower = (blockIdx.x > blockIdx.y);
    int ii = (lower ? (blockIdx.x - 1) : (blockIdx.y + gridDim.y));
    int jj = (lower ? (blockIdx.y    ) : (blockIdx.x + gridDim.y));

    ii *= NB;
    jj *= NB;

    float *A = matrix + ii+i + (jj+j)*lda;
    if ( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        __syncthreads();
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        float *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = *A;
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = *B;
        }
        __syncthreads();
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    stranspose_inplace_q transposes a square N-by-N matrix in-place.
    
    Same as stranspose_inplace, but adds queue argument.
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of rows & columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      REAL array, dimension (LDDA,N)
            The N-by-N matrix dA.
            On exit, dA(j,i) = dA_original(i,j), for 0 <= i,j < N.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_transpose
*******************************************************************************/
extern "C" void
magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( ldda < n )
        info = -3;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    dim3 threads( NB, NB );
    int nblock = magma_ceildiv( n, NB );
    
    // need 1/2 * (nblock+1) * nblock to cover lower triangle and diagonal of matrix.
    // block assignment differs depending on whether nblock is odd or even.
    if ( nblock % 2 == 1 ) {
        dim3 grid( nblock, (nblock+1)/2 );
        hipLaunchKernelGGL(stranspose_inplace_odd, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  n, dA, ldda );
    }
    else {
        dim3 grid( nblock+1, nblock/2 );
        hipLaunchKernelGGL(stranspose_inplace_even, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  n, dA, ldda );
    }
}
