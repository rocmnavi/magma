/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan  Tomov
       @author Mark  Gates
       @author Ahmad Abdelfattah

       @generated from src/zgetrf_gpu.cpp, normal z -> s, Mon Jul 15 16:57:17 2024

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    SGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is an expert API, exposing more controls to the end user.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative:  Factorize dA using GPU only mode.
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @param[in]
    nb      INTEGER
            The blocking size used during the factorization. nb > 0;
            Users with no specific preference of nb can call magma_get_sgetrf_nb()
            or magma_get_sgetrf_native_nb() to get the value of nb as determined
            by MAGMA's internal tuning.

    @param[in]
    recnb   INTEGER
            The blocking size used during the recursive panel factorization (0 < recnb <= nb);
            Users with no specific preference of recnb can set it to a fixed value of 32.

    @param[in,out]
    host_work  Workspace, allocated on host (CPU) memory. For faster CPU-GPU communication,
               user can allocate it as pinned memory using magma_malloc_pinned()

    @param[in,out]
    lwork_host   INTEGER pointer
                 The size of the workspace (host_work) in bytes
                 - lwork_host[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork_host. The workspace itself is not referenced, and no
                   factorization is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork_host.

    @param[in,out]
    device_work  Workspace, allocated on device (GPU) memory.

    @param[in,out]
    lwork_device   INTEGER pointer
                   The size of the workspace (device_work) in bytes
                   - lwork_device[0] < 0: a workspace query is assumed, the routine
                     calculates the required amount of workspace and returns
                     it in lwork_device. The workspace itself is not referenced, and no
                     factorization is performed.
                   - lwork_device[0] >= 0: the routine assumes that the user has provided
                     a workspace with the size in lwork_device.

    @param[in]
    events        magma_event_t array of size two
                  - created/destroyed by the user outside the routine
                  - Used to manage inter-stream dependencies

    @param[in]
    queues        magma_queue_t array of size two
                  - created/destroyed by the user outside the routine
                  - Used for concurrent kernel execution, if possible

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_sgetrf_expert_gpu_work(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info, magma_mode_t mode,
    magma_int_t nb, magma_int_t recnb,
    void* host_work,   magma_int_t *lwork_host,
    void* device_work, magma_int_t *lwork_device,
    magma_event_t events[2], magma_queue_t queues[2] )
{
    #define  dA(i_, j_) (dA  + (i_) + (size_t)(j_)*(size_t)ldda)
    #define dAT(i_, j_) (dAT + (j_) + (size_t)(i_)*(size_t)lddat)
    #define dAP(i_, j_) (dAP + (i_) + (size_t)(j_)*(size_t)maxm)

    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    magma_int_t iinfo;
    magma_int_t maxm, maxn, minmn;
    magma_int_t i, j, jb, rows, lddat, ldwork;
    magmaFloat_ptr dAT=NULL, dAP=NULL;
    float *work=NULL; // hybrid
    magma_int_t *dipiv=NULL, *dipivinfo=NULL, *dinfo=NULL; // native

    minmn = min( m, n );
    maxm  = magma_roundup( m, 32 );
    maxn  = magma_roundup( n, 32 );

    // calculate the required workspace in bytes
    magma_int_t h_workspace_bytes = 0;
    magma_int_t d_workspace_bytes = 0;
    lddat = (m == n) ? ldda : maxn;
    if (mode == MagmaHybrid) {
        if ( nb <= 1 || 4*nb >= n ) {
            h_workspace_bytes +=  m * n * sizeof(float); // all of A
        }
        else {
            ldwork = maxm;
            h_workspace_bytes += nb * ldwork * sizeof(float); // dAP on CPU
            d_workspace_bytes += nb * maxm * sizeof(float);   // dAP on GPU
            if( !(m == n) ) {
                d_workspace_bytes += lddat * maxm * sizeof(float); // separate memory for dAT
            }
        }
    }
    else {
        // native workspace
        d_workspace_bytes += nb * maxm * sizeof(float); // dAP
        d_workspace_bytes += (m + minmn + 1) * sizeof(magma_int_t);  // aux. pivot, pivot on device, & device info
        if( !(m == n) ) {
            d_workspace_bytes += lddat * maxm * sizeof(float); // separate memory for dAT
        }
    }

    // check for workspace query
    if( *lwork_host < 0 || *lwork_device < 0 ) {
        *lwork_host   = h_workspace_bytes;
        *lwork_device = d_workspace_bytes;
        *info  = 0;
        return *info;
    }

    *info = 0;
    /* Quick return if possible */
    if (m == 0 || n == 0) return *info;

    /* Check arguments */
    if (m < 0) {
        *info = -1;
    }
    else if (n < 0) {
        *info = -2;
    }
    else if (ldda < max(1,m)) {
        *info = -4;
    }
    else if (mode != MagmaHybrid && mode != MagmaNative) {
        *info = -7;
    }
    else if (nb < 1) {
        *info = -8;
    }
    else if (recnb < 1) {
        *info = -9;
    }
    else if( *lwork_host   < h_workspace_bytes ) {
        *info = -11;
    }
    else if( *lwork_device < d_workspace_bytes ) {
        *info = -13;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // assign pointers
    if( mode == MagmaHybrid ) {
        work = (float*)host_work;
        dAP  = (float*)device_work;
        dAT  = ( m == n ) ? dA : dAP + (nb * maxm);
    }
    else {
        work      = NULL;
        dAP       = (float*)device_work;
        dAT       = ( m == n ) ? dA : dAP + (nb * maxm);
        dipivinfo = ( m == n ) ? (magma_int_t*)(dAP + nb * maxm) :
                                 (magma_int_t*)(dAT + lddat * maxm);
        dipiv = dipivinfo + m;
        dinfo = dipiv + minmn;
        magma_memset_async(dinfo, 0, sizeof(magma_int_t), queues[0]);
    }

    // check for small sizes
    if ( nb <= 1 || 4*nb >= min(m,n) ) {
        if (mode == MagmaHybrid) {
            magma_sgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0]);
            lapackf77_sgetrf( &m, &n, work, &m, ipiv, info );
            magma_ssetmatrix( m, n, work, m, dA(0,0), ldda, queues[0]);
            //magma_free_cpu( work );  work=NULL;
            return *info;
        }
        else {
            // use non-transposed panel factorization for the whole matrix
            magma_sgetrf_recpanel_native( m, n, recnb, dA(0,0), ldda, dipiv, dipivinfo, dinfo, 0, events, queues[0], queues[1]);
            magma_igetvector_async( minmn, dipiv, 1, ipiv, 1, queues[0] );
            magma_igetvector_async( 1, dinfo, 1, info, 1, queues[0] );
            return *info;
        }
    }

    // square matrices can be done in place;
    // rectangular requires copy to transpose
    if ( m == n ) {
        magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[0] );
    }
    else {
        magmablas_stranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
    }

    // wait for transpose to finish
    if( mode == MagmaHybrid ) {
        magma_queue_sync( queues[0] );
    }
    else {
        magma_event_record( events[0], queues[0] );
        magma_queue_wait_event( queues[1], events[0] );
    }

    // main loop
    for( j=0; j < minmn-nb; j += nb ) {
        // get j-th panel from device
        magmablas_stranspose( nb, m-j, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
        magma_queue_sync( queues[1] );  // wait for transpose

        if (mode == MagmaHybrid) {
            magma_sgetmatrix_async( m-j, nb, dAP(0,0), maxm, work, ldwork, queues[0] );
        }

        if ( j > 0 ) {
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-(j+nb), nb,
                         c_one, dAT(j-nb, j-nb), lddat,
                                dAT(j-nb, j+nb), lddat, queues[1] );
            magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                         n-(j+nb), m-j, nb,
                         c_neg_one, dAT(j-nb, j+nb), lddat,
                                    dAT(j,    j-nb), lddat,
                         c_one,     dAT(j,    j+nb), lddat, queues[1] );
        }

        rows = m - j;
        if (mode == MagmaHybrid) {
            // do the cpu part
            magma_queue_sync( queues[0] );  // wait to get work
            lapackf77_sgetrf( &rows, &nb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j;

            // send j-th panel to device
            magma_ssetmatrix_async( m-j, nb, work, ldwork, dAP, maxm, queues[0] );

            for( i=j; i < j + nb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_slaswp( n, dAT(0,0), lddat, j + 1, j + nb, ipiv, 1, queues[1] );

            magma_queue_sync( queues[0] );  // wait to set dAP
        }
        else {
            // do the panel on the GPU
            magma_sgetrf_recpanel_native( rows, nb, recnb, dAP(0,0), maxm, dipiv+j, dipivinfo, dinfo, j, events, queues[0], queues[1]);
            adjust_ipiv( dipiv+j, nb, j, queues[0]);
            #ifdef SWP_CHUNK
            magma_igetvector_async( nb, dipiv+j, 1, ipiv+j, 1, queues[0] );
            #endif

            magma_queue_sync( queues[0] );  // wait for the pivot
            #ifdef SWP_CHUNK
            magmablas_slaswp( n, dAT(0,0), lddat, j + 1, j + nb, ipiv, 1, queues[1] );
            #else
            magma_slaswp_columnserial(n, dAT(0,0), lddat, j + 1, j + nb, dipiv, queues[1]);
            #endif
        }
        magmablas_stranspose( m-j, nb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );

        // do the small non-parallel computations (next panel update)
        if ( j + nb < minmn - nb ) {
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         nb, nb,
                         c_one, dAT(j, j   ), lddat,
                                dAT(j, j+nb), lddat, queues[1] );
            magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                         nb, m-(j+nb), nb,
                         c_neg_one, dAT(j,    j+nb), lddat,
                                    dAT(j+nb, j   ), lddat,
                         c_one,     dAT(j+nb, j+nb), lddat, queues[1] );
        }
        else {
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-(j+nb), nb,
                         c_one, dAT(j, j   ), lddat,
                                dAT(j, j+nb), lddat, queues[1] );
            magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                         n-(j+nb), m-(j+nb), nb,
                         c_neg_one, dAT(j,    j+nb), lddat,
                                    dAT(j+nb, j   ), lddat,
                         c_one,     dAT(j+nb, j+nb), lddat, queues[1] );
        }
    }

    jb = min( m-j, n-j );
    if ( jb > 0 ) {
        rows = m - j;

        magmablas_stranspose( jb, rows, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
        if (mode == MagmaHybrid) {
            magma_sgetmatrix( rows, jb, dAP(0,0), maxm, work, ldwork, queues[1] );

            // do the cpu part
            lapackf77_sgetrf( &rows, &jb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j;

            for( i=j; i < j + jb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_slaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );

            // send j-th panel to device
            magma_ssetmatrix( rows, jb, work, ldwork, dAP(0,0), maxm, queues[1] );
        }
        else {
            magma_sgetrf_recpanel_native( rows, jb, recnb, dAP(0,0), maxm, dipiv+j, dipivinfo, dinfo, j, events, queues[1], queues[0]);
            adjust_ipiv( dipiv+j, jb, j, queues[1]);
            #ifdef SWP_CHUNK
            magma_igetvector( jb, dipiv+j, 1, ipiv+j, 1, queues[1] );
            magmablas_slaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );
            #else
            magma_slaswp_columnserial(n, dAT(0,0), lddat, j + 1, j + jb, dipiv, queues[1]);
            #endif
        }

        magmablas_stranspose( rows, jb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );

        magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                     n-j-jb, jb,
                     c_one, dAT(j,j),    lddat,
                            dAT(j,j+jb), lddat, queues[1] );
    }

    if (mode == MagmaNative) {
        magma_igetvector_async( 1, dinfo, 1, info, 1, queues[0] );
        // copy the pivot vector to the CPU
        #ifndef SWP_CHUNK
        magma_igetvector_async(minmn, dipiv, 1, ipiv, 1, queues[1] );
        #endif
    }

    // undo transpose
    if ( m == n ) {
        magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[1] );
    }
    else {
        magmablas_stranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[1] );
    }

    return *info;
} /* magma_sgetrf_expert_gpu_work */


/***************************************************************************//**
    magma_sgetrf_gpu_expert is similar to magma_sgetrf_expert_gpu_work
    except that all workspaces/queues are handled internally
    @see magma_sgetrf_expert_gpu_work
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_sgetrf_gpu_expert(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info,
    magma_int_t nb, magma_mode_t mode)
{
    magma_device_t cdev;
    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_getdevice( &cdev );

    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);
    magma_int_t recnb = 32;

    // query workspace
    void *hwork = NULL, *dwork=NULL;
    magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
    magma_sgetrf_expert_gpu_work(
        m, n, NULL, ldda,
        NULL, info, mode, nb, recnb,
        NULL, lhwork, NULL, ldwork,
        events, queues );

    // alloc workspace
    if( lhwork[0] > 0 ) {
        magma_malloc_pinned( (void**)&hwork, lhwork[0] );
    }

    if( ldwork[0] > 0 ) {
        magma_malloc( (void**)&dwork, ldwork[0] );
    }

    magma_sgetrf_expert_gpu_work(
        m, n, dA, ldda, ipiv, info,
        mode, nb, recnb,
        hwork, lhwork, dwork, ldwork,
        events, queues );
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );

    // free workspace
    if( hwork != NULL ) magma_free_pinned( hwork );
    if( dwork != NULL ) magma_free( dwork );

    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    return *info;
}

/***************************************************************************//**
    magma_sgetrf_expert_gpu_work with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_sgetrf_expert_gpu_work
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_sgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_device_t cdev;
    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_getdevice( &cdev );

    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);

    magma_mode_t mode = MagmaHybrid;
    magma_int_t nb    = magma_get_sgetrf_nb( m, n );
    magma_int_t recnb = 32;

    // query workspace
    void *hwork = NULL, *dwork=NULL;
    magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
    magma_sgetrf_expert_gpu_work(
        m, n, NULL, ldda,
        NULL, info, mode, nb, recnb,
        NULL, lhwork, NULL, ldwork,
        events, queues );

    // alloc workspace
    if( lhwork[0] > 0 ) {
        magma_malloc_pinned( (void**)&hwork, lhwork[0] );
    }

    if( ldwork[0] > 0 ) {
        magma_malloc( (void**)&dwork, ldwork[0] );
    }

    magma_sgetrf_expert_gpu_work(
        m, n, dA, ldda, ipiv, info,
        mode, nb, recnb,
        hwork, lhwork, dwork, ldwork,
        events, queues );
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );

    // free workspace
    if( hwork != NULL ) magma_free_pinned( hwork );
    if( dwork != NULL ) magma_free( dwork );

    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    return *info;
} /* magma_sgetrf_gpu */

/***************************************************************************//**
    magma_sgetrf_expert_gpu_work with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_sgetrf_expert_gpu_work
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_sgetrf_native(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_device_t cdev;
    magma_queue_t queues[2];
    magma_event_t events[2];
    magma_getdevice( &cdev );

    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);

    magma_mode_t mode = MagmaNative;
    magma_int_t nb    = magma_get_sgetrf_native_nb( m, n );
    magma_int_t recnb = 32;

    // query workspace
    void *hwork = NULL, *dwork=NULL;
    magma_int_t lhwork[1] = {-1}, ldwork[1] = {-1};
    magma_sgetrf_expert_gpu_work(
        m, n, NULL, ldda,
        NULL, info, mode, nb, recnb,
        NULL, lhwork, NULL, ldwork,
        events, queues );

    // alloc workspace
    if( lhwork[0] > 0 ) {
        magma_malloc_pinned( (void**)&hwork, lhwork[0] );
    }

    if( ldwork[0] > 0 ) {
        magma_malloc( (void**)&dwork, ldwork[0] );
    }

    magma_sgetrf_expert_gpu_work(
        m, n, dA, ldda, ipiv, info,
        mode, nb, recnb,
        hwork, lhwork, dwork, ldwork,
        events, queues );
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );

    // free workspace
    if( hwork != NULL ) magma_free_pinned( hwork );
    if( dwork != NULL ) magma_free( dwork );

    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    return *info;
} /* magma_sgetrf_native */
