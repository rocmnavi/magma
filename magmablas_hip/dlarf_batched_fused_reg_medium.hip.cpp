/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zlarf_batched_fused_reg_medium.hip.cpp, normal z -> d, Fri Aug 25 13:17:10 2023
*/

#include <hip/hip_runtime.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "dlarf_batched_fused.hip.hpp"
#include "batched_kernel_param.h"

#define PRECISION_d

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dlarf_fused_reg_medium_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    double **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m32 = magma_roundup(m, 32);

    if (m32 < nb)
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
        case 12: arginfo = magma_dlarf_fused_reg_NB_batched<384>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 13: arginfo = magma_dlarf_fused_reg_NB_batched<416>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 14: arginfo = magma_dlarf_fused_reg_NB_batched<448>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 15: arginfo = magma_dlarf_fused_reg_NB_batched<480>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 16: arginfo = magma_dlarf_fused_reg_NB_batched<512>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 17: arginfo = magma_dlarf_fused_reg_NB_batched<544>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 18: arginfo = magma_dlarf_fused_reg_NB_batched<576>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 19: arginfo = magma_dlarf_fused_reg_NB_batched<608>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 20: arginfo = magma_dlarf_fused_reg_NB_batched<640>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 21: arginfo = magma_dlarf_fused_reg_NB_batched<672>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 22: arginfo = magma_dlarf_fused_reg_NB_batched<704>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 23: arginfo = magma_dlarf_fused_reg_NB_batched<736>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}
