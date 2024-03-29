#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zlarf_batched_fused.hip.hpp, normal z -> d, Fri Aug 25 13:18:27 2023
*/

////////////////////////////////////////////////////////////////////////////////
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define sA(i,j)               sA[(j) * slda + (i)]
#define sV(i,j)               sV[(j) * sldv + (i)]
#define sT(i,j)               sT[(j) * sldt + (i)]
#define NTCOL(M)             ((M > 32) ? 1 : 2)

////////////////////////////////////////////////////////////////////////////////
//              DLARF fused register kernel
////////////////////////////////////////////////////////////////////////////////
template<int M32, int NB, int TPC>
__global__
__launch_bounds__(M32*NTCOL(M32))
void
dlarf_fused_reg_kernel_batched(
    int m, int n, int ib,
    double **dA_array, int Ai, int Aj, int ldda,
    double **dV_array, int Vi, int Vj, int lddv,
    double **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only, magma_int_t batchCount )
{
    HIP_DYNAMIC_SHARED( double, zdata)

    // if check_launch_only = 1, then return immediately
    // this is only to check if the kernel has been launched
    // successfully
    if(check_launch_only == 1) return;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ty_ = tx / TPC;
    const int tx_ = tx % TPC;
    const int nty = blockDim.y;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    double* dA   = dA_array[batchid] + Aj * ldda + Ai;
    double* dV   = dV_array[batchid] + Vj * lddv + Vi;
    double* dtau = dtau_array[batchid] + taui;

    double rA[NB] = {MAGMA_D_ZERO};
    const int slda = SLDA(M32);
    const int sldv = SLDA(M32);
    const int sldt = SLDA(TPC);

    // shared memory pointers
    double* sV   = (double*)(zdata);
    double* sA   = sV + (nty * sldv * NB);
    double* sT   = sA + (nty * slda * NB);
    double* stau = sT + (nty * sldt * NB);
    sV    += ty * sldv * NB;
    sA    += ty * slda * NB;
    sT    += ty * sldt * NB;
    stau  += ty * NB;

    double zsum;
    int i, iib;

    // init sA,sV to zero
    #pragma unroll
    for(int j = 0; j < NB; j++) {
        sA(tx,j) = MAGMA_D_ZERO;
        sV(tx,j) = MAGMA_D_ZERO;
    }

    // read tau and init diag(sV)
    if(tx < ib) {
        stau[tx]  = dtau[tx];
        sV(tx,tx) = MAGMA_D_ONE; // does not need a sync before it
    }

    // read into rA and sV
    if( tx < m ) {
        for(int j = 0; j < ib; j++) {
            //rA[j]    = dA[ j * ldda + tx ];
            sV(tx,j) = (tx > j) ? dV[j * lddv + tx] : sV(tx,j);
        }
    }

    //////////// main loop ////////////////
    for(iib = 0; iib < (n/NB)*NB; iib+=NB) {
        // read A
        if(tx < m) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                rA[j]    = dA[ j * ldda + tx ];
            }
        }

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // compute v' * rA -> sA
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                sA(tx,jj) = MAGMA_D_CONJ( sV(tx,j) ) * rA[jj];
            }
            __syncthreads();

            // reduce (1-of-2)
            zsum = MAGMA_D_ZERO;
            if(ty_ < NB) {
                #pragma unroll
                for(i = 0; i < M32-TPC; i+=TPC) {
                    zsum += sA(tx_+i,ty_);
                }

                if(tx_ < M32-i) {
                    zsum += sA(tx_+i,ty_);
                }

                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_D_ZERO;
            if(tx < NB) {
                #pragma unroll
                for(i = 0; i < TPC; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = MAGMA_D_CONJ( stau[j] ) * zsum;
            }
            __syncthreads();

            // rank update
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                rA[jj] -= sV(tx,j) * sT(0,jj);
            }
        }    // end of apply loop

        // write rA
        if(tx < m) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                dA[ j * ldda + tx ] = rA[j];
            }
        }

        // advance dA
        dA += NB*ldda;
    }    // end of main loop

    //////////// cleanup section ////////////////
    if(n - iib > 0) {
        int nn = n - iib;
        // read A
        if(tx < m) {
            for(int j = 0; j < nn; j++) {
                sA(tx,j) = dA[ j * ldda + tx ];
            }
        }
        __syncthreads();

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // reduce (1-of-2)
            zsum = MAGMA_D_ZERO;
            if(ty_ < nn) {
                #pragma unroll
                for(i = 0; i < M32-TPC; i+=TPC) {
                    zsum += sA(tx_+i,ty_) * MAGMA_D_CONJ( sV(tx_+i,j) );
                }

                if(tx_ < M32-i) {
                    zsum += sA(tx_+i,ty_) * MAGMA_D_CONJ( sV(tx_+i,j) );
                }

                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_D_ZERO;
            if(tx < nn) {
                #pragma unroll
                for(i = 0; i < TPC; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = MAGMA_D_CONJ( stau[j] ) * zsum;
            }
            __syncthreads();

            // rank update
            for(int jj = 0; jj < nn; jj++) {
                sA(tx,jj) -= sV(tx,j) * sT(0,jj);
            }
            __syncthreads();

        }    // end of apply loop

        // write rA
        if(tx < m) {
            for(int j = 0; j < nn; j++) {
                dA[ j * ldda + tx ] = sA(tx,j);
            }
        }
    }    // end of cleanup section

}

////////////////////////////////////////////////////////////////////////////////
//              DLARF fused register kernel driver
////////////////////////////////////////////////////////////////////////////////
template<int M32, int NB>
static magma_int_t
magma_dlarf_fused_reg_kernel_driver_batched(
    magma_int_t m, magma_int_t n, magma_int_t ib,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    double **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;
    magma_int_t nthreads = M32;
    const magma_int_t ntcol = NTCOL(M32);
    const magma_int_t TPC   = M32 / NB;

    magma_int_t shmem = 0;
    shmem += SLDA(M32) * NB * sizeof(double);  // sA
    shmem += SLDA(M32) * NB * sizeof(double);  // sV
    shmem += SLDA(TPC) * NB * sizeof(double);  // sT
    shmem += NB             * sizeof(double);  // stau
    shmem *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads( nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    hipDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(dlarf_fused_reg_kernel_batched<M32, NB, TPC>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    //if(check_launch_only == 1) return arginfo;
    void *kernel_args[] = {&m, &n, &ib, &dA_array, &Ai, &Aj, &ldda, &dV_array, &Vi, &Vj, &lddv, &dtau_array, &taui, &check_launch_only, &batchCount};
    hipError_t e = hipLaunchKernel((void*)dlarf_fused_reg_kernel_batched<M32, NB, TPC>, grid, threads, kernel_args, shmem, queue->hip_stream());
    if( e != hipSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, hipGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
//              DLARF fused register kernel driver
//              instantiations based on n
////////////////////////////////////////////////////////////////////////////////
template<int M32>
static magma_int_t
magma_dlarf_fused_reg_NB_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    double **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    switch(nb) {
        case 1: arginfo = magma_dlarf_fused_reg_kernel_driver_batched<M32, 1>( m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 2: arginfo = magma_dlarf_fused_reg_kernel_driver_batched<M32, 2>( m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 4: arginfo = magma_dlarf_fused_reg_kernel_driver_batched<M32, 4>( m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 8: arginfo = magma_dlarf_fused_reg_kernel_driver_batched<M32, 8>( m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        #if defined(MAGMA_HAVE_CUDA) && !defined(PRECISION_z)
        case 16: arginfo = magma_dlarf_fused_reg_kernel_driver_batched<M32,16>( m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        #endif
        default: arginfo = -100;
    }
    return arginfo;
}
