/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah

       @generated from magmablas/zlarf_batched_fused_reg_tall.cu, normal z -> s, Fri Aug 25 13:18:51 2023
*/

#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "slarf_batched_fused.cuh"
#include "batched_kernel_param.h"

#define PRECISION_s

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_slarf_fused_reg_tall_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    float** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    float **dtau_array, magma_int_t taui,
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
        case 24: arginfo = magma_slarf_fused_reg_NB_batched<768>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 25: arginfo = magma_slarf_fused_reg_NB_batched<800>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 26: arginfo = magma_slarf_fused_reg_NB_batched<832>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 27: arginfo = magma_slarf_fused_reg_NB_batched<864>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 28: arginfo = magma_slarf_fused_reg_NB_batched<896>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 29: arginfo = magma_slarf_fused_reg_NB_batched<928>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 30: arginfo = magma_slarf_fused_reg_NB_batched<960>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 31: arginfo = magma_slarf_fused_reg_NB_batched<992>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 32: arginfo = magma_slarf_fused_reg_NB_batched<1024>(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}
