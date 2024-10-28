/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zgeqr2_batched_fused_reg.hip.cpp, normal z -> s, Mon Oct 28 11:12:18 2024
*/

#include <hip/hip_runtime.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "sgeqr2_batched_fused.hip.hpp"
#include "batched_kernel_param.h"

#define PRECISION_s

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_sgeqr2_fused_reg_batched(
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

    magma_int_t m32 = magma_roundup(m,32);
    if( m32 >= 768 ) {
        arginfo = magma_sgeqr2_fused_reg_tall_batched( m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue );
    }
    else if( m32 >= 384 ) {
        arginfo = magma_sgeqr2_fused_reg_medium_batched( m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue );
    }
    else {
        switch( magma_ceildiv(m,32) ) {
            case  1: arginfo = magma_sgeqr2_fused_reg_N_batched< 32>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  2: arginfo = magma_sgeqr2_fused_reg_N_batched< 64>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  3: arginfo = magma_sgeqr2_fused_reg_N_batched< 96>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  4: arginfo = magma_sgeqr2_fused_reg_N_batched<128>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  5: arginfo = magma_sgeqr2_fused_reg_N_batched<160>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  6: arginfo = magma_sgeqr2_fused_reg_N_batched<192>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  7: arginfo = magma_sgeqr2_fused_reg_N_batched<224>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  8: arginfo = magma_sgeqr2_fused_reg_N_batched<256>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case  9: arginfo = magma_sgeqr2_fused_reg_N_batched<288>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case 10: arginfo = magma_sgeqr2_fused_reg_N_batched<320>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            case 11: arginfo = magma_sgeqr2_fused_reg_N_batched<352>(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
            default: arginfo = -100;
        }
    }
    return arginfo;
}
