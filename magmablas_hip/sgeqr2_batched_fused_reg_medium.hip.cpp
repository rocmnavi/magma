/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zgeqr2_batched_fused_reg_medium.hip.cpp, normal z -> s, Fri Aug 25 13:17:07 2023
*/

#include <hip/hip_runtime.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "sgeqr2_batched_fused.hip.hpp"
#include "batched_kernel_param.h"

#define PRECISION_s

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_sgeqr2_fused_reg_medium_batched(
    magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float **dtau_array, magma_int_t taui,
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
        case 12: arginfo = magma_sgeqr2_fused_reg_N_batched<384>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 13: arginfo = magma_sgeqr2_fused_reg_N_batched<416>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 14: arginfo = magma_sgeqr2_fused_reg_N_batched<448>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 15: arginfo = magma_sgeqr2_fused_reg_N_batched<480>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 16: arginfo = magma_sgeqr2_fused_reg_N_batched<512>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 17: arginfo = magma_sgeqr2_fused_reg_N_batched<544>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 18: arginfo = magma_sgeqr2_fused_reg_N_batched<576>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 19: arginfo = magma_sgeqr2_fused_reg_N_batched<608>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 20: arginfo = magma_sgeqr2_fused_reg_N_batched<640>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 21: arginfo = magma_sgeqr2_fused_reg_N_batched<672>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 22: arginfo = magma_sgeqr2_fused_reg_N_batched<704>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 23: arginfo = magma_sgeqr2_fused_reg_N_batched<736>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}
