/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zgeqr2_batched_fused_reg_tall.hip.cpp, normal z -> d, Fri Aug 25 13:17:08 2023
*/

#include <hip/hip_runtime.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "dgeqr2_batched_fused.hip.hpp"
#include "batched_kernel_param.h"

#define PRECISION_d

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dgeqr2_fused_reg_tall_batched(
    magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;

    if (m < n)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return arginfo;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    switch( magma_ceildiv(m,32) ) {
        case 24: arginfo = magma_dgeqr2_fused_reg_N_batched<768>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 25: arginfo = magma_dgeqr2_fused_reg_N_batched<800>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 26: arginfo = magma_dgeqr2_fused_reg_N_batched<832>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 27: arginfo = magma_dgeqr2_fused_reg_N_batched<864>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 28: arginfo = magma_dgeqr2_fused_reg_N_batched<896>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 29: arginfo = magma_dgeqr2_fused_reg_N_batched<928>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 30: arginfo = magma_dgeqr2_fused_reg_N_batched<960>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 31: arginfo = magma_dgeqr2_fused_reg_N_batched<992>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 32: arginfo = magma_dgeqr2_fused_reg_N_batched<1024>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}
