#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @generated from sparse_hip/blas/magma_zmatrixtools_gpu.hip.cpp, normal z -> c, Fri Aug 25 13:17:55 2023

*/
#include "magmasparse_internal.h"

#define PRECISION_c

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#ifdef MAGMA_HAVE_HIP
  #define hipblasComplex hipFloatComplex
#endif



__global__ void 
magma_cvalinit_kernel(  
    const magma_int_t num_el, 
    magmaFloatComplex_ptr dval) 
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    if (k < num_el) {
        dval[k] = zero;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dval        magmaFloatComplex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_cvalinit_gpu(
    magma_int_t num_el,
    magmaFloatComplex_ptr dval,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    dim3 grid(dimgrid1, dimgrid2, dimgrid3);
    dim3 block(blocksize1, blocksize2, 1);
    hipLaunchKernelGGL(magma_cvalinit_kernel, dim3(grid), dim3(block), 0, queue->hip_stream() , num_el, dval);

    return MAGMA_SUCCESS;
}




__global__ void 
magma_cindexinit_kernel(  
    const magma_int_t num_el, 
    magmaIndex_ptr dind) 
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_el) {
        dind[k] = 0;
    }
}


/**
    Purpose
    -------
    
    Initializes a device array with zero. 

    Arguments
    ---------

    @param[in]
    num_el      magma_int_t
                size of array

    @param[in,out]
    dind        magmaIndex_ptr
                array to initialize
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_cindexinit_gpu(
    magma_int_t num_el,
    magmaIndex_ptr dind,
    magma_queue_t queue)
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid1 = magma_ceildiv(num_el, blocksize1);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    dim3 grid(dimgrid1, dimgrid2, dimgrid3);
    dim3 block(blocksize1, blocksize2, 1);
    hipLaunchKernelGGL(magma_cindexinit_kernel, dim3(grid), dim3(block), 0, queue->hip_stream() , num_el, dind);

    return MAGMA_SUCCESS;
}


__global__ void 
magma_cmatrixcup_count(  
    const magma_int_t num_rows,
    const magma_index_t* A_row,
    const magma_index_t* A_col,
    const magma_index_t* B_row,
    const magma_index_t* B_col,
    magma_index_t* inserted)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        int add = 0;
        int a = A_row[row];
        int b = B_row[row];
        int enda = A_row[ row+1 ];
        int endb = B_row[ row+1 ]; 
        int acol;
        int bcol;
        if (a<enda && b<endb) {
            do{
                acol = A_col[ a ];
                bcol = B_col[ b ];
                
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    add++;
                    a++;
                }
                else {
                    add++;
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                add++;
                b++;
            }while(b<endb);            
        }
        inserted[ row ] = add; 
    }
}


__global__ void 
magma_cmatrixcup_fill(  
    const magma_int_t num_rows,
    const magma_index_t* A_row,
    const magma_index_t* A_col,
    const magmaFloatComplex* A_val,
    const magma_index_t* B_row,
    const magma_index_t* B_col,
    const magmaFloatComplex* B_val,
    magma_index_t* U_row,
    magma_index_t* U_rowidx,
    magma_index_t* U_col,
    magmaFloatComplex* U_val)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        int add = 0;
        int offset = U_row[row];
        int a = A_row[row];
        int b = B_row[row];
        int enda = A_row[ row+1 ];
        int endb = B_row[ row+1 ]; 
        int acol;
        int bcol;
        if (a<enda && b<endb) {
            do{
                acol = A_col[ a ];
                bcol = B_col[ b ];
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    U_col[ offset + add ] = acol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = A_val[ a ];
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    U_col[ offset + add ] = acol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = A_val[ a ];
                    add++;
                    a++;
                }
                else {
                    U_col[ offset + add ] = bcol;
                    U_rowidx[ offset + add ] = row;
                    U_val[ offset + add ] = B_val[ b ];
                    add++;
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A_col[ a ];
                U_col[ offset + add ] = acol;
                U_rowidx[ offset + add ] = row;
                U_val[ offset + add ] = A_val[ a ];
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                bcol = B_col[ b ];
                U_col[ offset + add ] = bcol;
                U_rowidx[ offset + add ] = row;
                U_val[ offset + add ] = B_val[ b ];
                add++;
                b++;
            }while(b<endb);            
        }
    }
}
    
    
    

/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.
    
    This is the GPU version of the operation.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                Input matrix 1.

    @param[in]
    B           magma_c_matrix
                Input matrix 2.

    @param[out]
    U           magma_c_matrix*
                U = A \cup B. If both matrices have a nonzero value 
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/

extern "C" magma_int_t
magma_cmatrix_cup_gpu(
    magma_c_matrix A,
    magma_c_matrix B,
    magma_c_matrix *U,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    assert(A.num_rows == B.num_rows);
    magma_int_t num_rows = A.num_rows;
    U->num_rows = num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_DEV;
   

    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid11 = magma_ceildiv(num_rows, blocksize1 );
    int dimgrid12 = 1;
    int dimgrid13 = 1;
    dim3 grid1(dimgrid11, dimgrid12, dimgrid13 );
    dim3 block1(blocksize1, blocksize2, 1 );
    
    magmaIndex_ptr inserted = NULL;
    CHECK(magma_index_malloc(&U->drow, num_rows+1));
    CHECK(magma_index_malloc(&inserted, num_rows));
    CHECK(magma_cindexinit_gpu(num_rows, inserted, queue));
    
    hipLaunchKernelGGL(magma_cmatrixcup_count, dim3(grid1), dim3(block1), 0, queue->hip_stream(), num_rows, A.drow, A.dcol, B.drow, B.dcol, inserted);
    
    CHECK(magma_cget_row_ptr(num_rows, &U->nnz, inserted, U->drow, queue));
    

    CHECK(magma_cmalloc(&U->dval, U->nnz));
    CHECK(magma_index_malloc(&U->drowidx, U->nnz));
    CHECK(magma_index_malloc(&U->dcol, U->nnz));
    
    hipLaunchKernelGGL(magma_cmatrixcup_fill, dim3(grid1), dim3(block1), 0, queue->hip_stream(), num_rows, A.drow, A.dcol, A.dval, B.drow, B.dcol, B.dval,
        U->drow, U->drowidx, U->dcol, U->dval);
    
cleanup:
    magma_free(inserted);
    return info;
}





/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.
    
    This is the GPU version of the operation.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                Input matrix 1.

    @param[in]
    B           magma_c_matrix
                Input matrix 2.

    @param[out]
    U           magma_c_matrix*
                U = A \cup B. If both matrices have a nonzero value 
                in the same location, the value of A is used.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
*******************************************************************************/

extern "C" magma_int_t
magma_ccsr_sort_gpu(
    magma_c_matrix *A,
    magma_queue_t queue)
{   
    magma_int_t info = 0;
    hipsparseHandle_t handle=NULL;
    hipsparseMatDescr_t descrA=NULL;
    
    magmaFloatComplex_ptr tmp=NULL, csrVal_sorted=NULL;
    char* pBuffer;
    int *P;
    size_t pBufferSizeInBytes;
    
    CHECK_CUSPARSE( hipsparseCreate( &handle ));
    CHECK_CUSPARSE( hipsparseSetStream( handle, queue->hip_stream() ));
    CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( hipsparseSetMatType( descrA, 
        HIPSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( hipsparseSetMatDiagType( descrA, 
        HIPSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, 
        HIPSPARSE_INDEX_BASE_ZERO ));
    
    CHECK(magma_cmalloc(&csrVal_sorted, A->nnz));
   
    // step 1: allocate buffer
    hipsparseXcsrsort_bufferSizeExt(handle, A->num_rows, A->num_cols, 
        A->nnz, A->drow, A->dcol, &pBufferSizeInBytes);
    hipMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);
    
    // step 2: setup permutation vector P to identity
    hipMalloc( (void**)&P, sizeof(int)*A->nnz);
    hipsparseCreateIdentityPermutation(handle, A->nnz, P);
    
    // step 3: sort CSR format
    hipsparseXcsrsort(handle, A->num_rows, A->num_cols, A->nnz, 
        descrA, A->drow, A->dcol, P, pBuffer);
    
    // step 4: gather sorted csrVal
#if CUDA_VERSION >= 12000
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, A->nnz, A->nnz,
                                        P, csrVal_sorted,
                                        CUSPARSE_INDEX_32I,
                                        HIPSPARSE_INDEX_BASE_ZERO, HIPBLAS_C_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, A->nnz, A->dval, HIPBLAS_C_64F) );
    CHECK_CUSPARSE( cusparseGather(handle, vec_values, vec_permutation) );
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) );
#else
    hipsparseCgthr(handle, A->nnz, (hipblasComplex*)A->dval, (hipblasComplex*)csrVal_sorted, P, 
        HIPSPARSE_INDEX_BASE_ZERO);
#endif

    SWAP(A->dval, csrVal_sorted);
    
cleanup:
    hipsparseDestroyMatDescr( descrA );
    hipsparseDestroy( handle );
    magma_free(csrVal_sorted);

    return info;
}