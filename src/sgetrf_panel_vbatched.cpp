/*
   -- MAGMA (version 2.7.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date August 2023

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from src/zgetrf_panel_vbatched.cpp, normal z -> s, Fri Aug 25 13:16:48 2023
*/
#include "magma_internal.h"

#define PRECISION_s
#define SGETRF2_VBATCHED_PAR_SWAP

// always assume for every matrix m >= n
// then max_minmn = max_n
magma_int_t
magma_sgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue)
{
#define dA_array(i,j)    dA_array, i, j
#define dipiv_array(i)   dipiv_array, i

    if( max_n <= min_recpnb ) {
        magma_sgetf2_vbatched(
            m, n, minmn,
            max_m, max_n, max_minmn, max_mxn,
            dA_array, Ai, Aj, ldda,
            dipiv_array, info_array,
            gbstep, batchCount, queue);
    }
    else {
        magma_int_t max_n1 = max( min_recpnb, max_n / 2);
        magma_int_t max_n2 = max_n - max_n1;
        magma_int_t new_max_minmn = max_m * max_n1;
        // panel
        magma_sgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m, max_n1, max_n1, new_max_minmn, min_recpnb,
            dA_array(Ai, Aj), ldda,
            dipiv_array, Ai, dpivinfo_array,
            info_array, gbstep, batchCount, queue);

        // swap right
        #ifdef SGETRF2_VBATCHED_PAR_SWAP
        setup_pivinfo_vbatched(dpivinfo_array, Ai, dipiv_array, Ai, m, n, max_m, max_n1, batchCount, queue);
        magma_slaswp_right_rowparallel_vbatched(
            max_n2,
            m, n,
            dA_array(Ai, Aj+max_n1), ldda,
            0, max_n1,
            dpivinfo_array, Ai,
            batchCount, queue);
        #else
        magma_slaswp_right_rowserial_vbatched(
            max_n2,
            m, n,
            dA_array(Ai, Aj+max_n1), ldda,
            dipiv_array(Ai),
            0, max_n1,
            batchCount, queue);
        #endif

        // trsm
        magmablas_strsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            max_n1, max_n2, m, n, MAGMA_S_ONE,
            dA_array(Ai, Aj       ), ldda,
            dA_array(Ai, Aj+max_n1), ldda,
            batchCount, queue );

        // gemm
        magmablas_sgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-max_n1, max_n2, max_n1,
            m, n, minmn,
            MAGMA_S_NEG_ONE, dA_array(Ai+max_n1, Aj       ), ldda,
                             dA_array(Ai       , Aj+max_n1), ldda,
            MAGMA_S_ONE,     dA_array(Ai+max_n1, Aj+max_n1), ldda,
            batchCount, queue );

        // panel 2
        new_max_minmn = (max_m-max_n1) * max_n2;
        magma_sgetrf_recpanel_vbatched(
            m, n, minmn,
            max_m-max_n1, max_n2, max_n2, new_max_minmn, min_recpnb,
            dA_array(Ai+max_n1, Aj+max_n1), ldda,
            dipiv_array, Ai+max_n1, dpivinfo_array,
            info_array, gbstep+max_n1, batchCount, queue);

        // swap left
        #ifdef SGETRF2_VBATCHED_PAR_SWAP
        setup_pivinfo_vbatched(dpivinfo_array, Ai+max_n1, dipiv_array, Ai+max_n1, m, n, max_m-max_n1, max_n2, batchCount, queue);
        magma_slaswp_left_rowparallel_vbatched(
            max_n1,
            m, n, dA_array(Ai+max_n1, Aj), ldda,
            0, max_n2,
            dpivinfo_array, Ai+max_n1,
            batchCount, queue);
        #else
        magma_slaswp_left_rowserial_vbatched(
            max_n1,
            m, n, dA_array(Ai+max_n1, Aj), ldda,
            dipiv_array(Ai+max_n1),
            0, max_n2,
            batchCount, queue);
        #endif

        // adjust pivot
        adjust_ipiv_vbatched(dipiv_array, Ai+max_n1, minmn, max_n2, max_n1, batchCount, queue);
    }

    return 0;

#undef dA_array
#undef dipiv_array
}
