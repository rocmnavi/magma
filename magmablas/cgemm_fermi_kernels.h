#ifndef CGEMM_FERMI_KERNELS_H
#define CGEMM_FERMI_KERNELS_H

/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       See [zcds]gemm_fermi.cu for description of related files.
*/
#include "magma_internal.h"

// =============================================================================

#define COMPLEX
//#undef DOUBLE
#ifdef MAGMA_HAVE_CUDA
  #define TEXTURE_1D
#endif

#include "gemm_stencil_defs.h"

// =============================================================================
// currently, CPU driver assumes all transpose versions have same DIM_X, DIM_Y

// size of thread block for calculating C (innermost loop)
#define DIM_X  16
#define DIM_Y  16


// =============================================================================
// A x B
#if defined(MAGMA_HAVE_CUDA)
    // size of work for a thread block
    #define BLK_M_nn  64
    #define BLK_N_nn  64

    #define BLK_K  16

    // size of thread block for reading A (dev->regs->shmem)
    #define DIM_XA 32
    #define DIM_YA 8

    // size of thread block for reading B (dev->regs->shmem)
    #define DIM_XB  16
    #define DIM_YB  16
#else
    // size of work for a thread block
    #define BLK_M_nn  64
    #define BLK_N_nn  32

    #define BLK_K  8

    // size of thread block for reading A (dev->regs->shmem)
    #define DIM_XA 32
    #define DIM_YA 8

    // size of thread block for reading B (dev->regs->shmem)
    #define DIM_XB  8
    #define DIM_YB  32
#endif

#undef  version
#define version trans_nn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A x B^T
// size of work for a thread block
#define BLK_M_nt  64
#define BLK_N_nt  64

#define BLK_M_nc  64
#define BLK_N_nc  64

#define BLK_K  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#undef  version
#define version trans_nt
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_nc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A^T x B^T
// size of work for a thread block
#define BLK_M_tt  64
#define BLK_N_tt  64

#define BLK_M_tc  64
#define BLK_N_tc  64

#define BLK_M_ct  64
#define BLK_N_ct  64

#define BLK_M_cc  64
#define BLK_N_cc  64

#define BLK_K  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 32
#define DIM_YB 8

#undef  version
#define version trans_tt
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_tc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_ct
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_cc
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


// =============================================================================
// A^T x B
// size of work for a thread block
#define BLK_M_tn  64
#define BLK_N_tn  64

#define BLK_M_cn  64
#define BLK_N_cn  64

#define BLK_K  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#undef  version
#define version trans_tn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef  version
#define version trans_cn
#include "gemm_stencil.cuh"
#include "gemm_kernel.cuh"

#undef COMPLEX

#endif // CGEMM_FERMI_KERNELS_H
