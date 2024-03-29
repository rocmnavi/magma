/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Tingxing Dong
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_d

#include "gemv_template_kernel_vbatched.hip.hpp"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_dgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue)
{
    if ( trans == MagmaNoTrans ) {                                                   
        if (max(max_m, max_n) <= 96) { // small size                         
            if (max_m < max_n) { // Fat or square matrix
                if ( max_m <= 16) 
                {
                    gemvn_template_vbatched<double, version(N, 72)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if ( max_m <= 32)            
                {
                    gemvn_template_vbatched<double, version(N, 100)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if ( max_m <= 64)            
                {
                    gemvn_template_vbatched<double, version(N, 122)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<double, version(N, 135)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }  
            } else {   // Tall or square matrix
                if ( max_n <= 16) 
                {    
                    gemvn_template_vbatched<double, version(N, 128)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if ( max_n <= 64) 
                {
                    gemvn_template_vbatched<double, version(N, 132)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }  
                else 
                {
                    gemvn_template_vbatched<double, version(N, 135)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }  
            }
        }
        else { // big size
            if (max_m < max_n) { // Fat matrix
                if (max_m <= 8)
                {
                    gemvn_template_vbatched<double, version(N, 79)>              
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 16)
                {
                    gemvn_template_vbatched<double, version(N, 70)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 32)
                {
                    gemvn_template_vbatched<double, version(N, 104)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 32)
                {
                    gemvn_template_vbatched<double, version(N, 124)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<double, version(N, 135)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
            else { // (m > n) Tall matrix
                if (max_m <= 256)
                {
                    gemvn_template_vbatched<double, version(N, 137)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<double, version(N, 140)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
        }// big size        
    }
    else {
        if (max(max_m, max_n) <= 96) { // small size                              
            if (max_m <= 16)
            {
                gemvc_template_vbatched<double, version(T, 42)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
            }
            else
            {
                gemvc_template_vbatched<double, version(T, 47)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
            }
        } else { // big size
            if (max_m <= max_n) { //  Fat or square matrix
                if (max_m <= 64)
                {
                    gemvc_template_vbatched<double, version(T, 47)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvc_template_vbatched<double, version(T, 91)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            } else { // (m > n) Tall matrix
                if (max_n <= 64)
                {
                    gemvc_template_vbatched<double, version(T, 90)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvc_template_vbatched<double, version(T, 91)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
        }
    }
}
