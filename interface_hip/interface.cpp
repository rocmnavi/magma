/*
    -- MAGMA (version 2.7.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2023

       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <map>

#if __cplusplus >= 201103  // C++11 standard
#include <mutex>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(MAGMA_WITH_ACML)
#include <acml.h>
#endif

#include <hip/hip_runtime.h>

// defining MAGMA_LAPACK_H is a hack to NOT include magma_lapack.h
// via magma_internal.h here, since it conflicts with acml.h and we don't
// need lapack here, but we want acml.h for the acmlversion() function.
#define MAGMA_LAPACK_H

#include "magma_internal.h"
#include "error.h"

#define MAX_BATCHCOUNT    (65534)

#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)

#ifdef DEBUG_MEMORY
// defined in alloc.cpp
extern std::map< void*, size_t > g_pointers_dev;
extern std::map< void*, size_t > g_pointers_cpu;
extern std::map< void*, size_t > g_pointers_pin;
#endif

// -----------------------------------------------------------------------------
// prototypes
extern "C" void
magma_warn_leaks( const std::map< void*, size_t >& pointers, const char* type );


// -----------------------------------------------------------------------------
// constants

// bit flags
enum {
    own_none     = 0x0000,
    own_stream   = 0x0001,
    own_cublas   = 0x0002,
    own_cusparse = 0x0004,
    own_opencl   = 0x0008,
    own_hip      = 0x0010,
    own_hipblas  = 0x0020,
    own_hipsparse= 0x0040
};


// -----------------------------------------------------------------------------
// globals
#if __cplusplus >= 201103  // C++11 standard
    static std::mutex g_mutex;
#else
    // without C++11, wrap pthread mutex
    class PthreadMutex {
    public:
        PthreadMutex()
        {
            int err = pthread_mutex_init( &mutex, NULL );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_init failed: %d\n", err );
            }
        }

        ~PthreadMutex()
        {
            int err = pthread_mutex_destroy( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_destroy failed: %d\n", err );
            }
        }

        void lock()
        {
            int err = pthread_mutex_lock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_lock failed: %d\n", err );
            }
        }

        void unlock()
        {
            int err = pthread_mutex_unlock( &mutex );
            if ( err ) {
                fprintf( stderr, "pthread_mutex_unlock failed: %d\n", err );
            }
        }

    private:
        pthread_mutex_t mutex;
    };

    static PthreadMutex g_mutex;
#endif

// count of (init - finalize) calls
static int g_init = 0;

#ifndef MAGMA_NO_V1
    magma_queue_t* g_null_queues = NULL;

    #ifdef HAVE_PTHREAD_KEY
    pthread_key_t g_magma_queue_key;
    #else
    magma_queue_t g_magma_queue = NULL;
    #endif
#endif // MAGMA_NO_V1


// -----------------------------------------------------------------------------
// subset of the CUDA device properties, set by magma_init()
struct magma_device_info
{
    size_t memory;
    size_t shmem_block;      // maximum shared memory per thread block in bytes
    size_t shmem_multiproc;  // maximum shared memory per multiprocessor in bytes
    magma_int_t cuda_arch;
    magma_int_t multiproc_count;    // number of multiprocessors
};

int g_magma_devices_cnt = 0;
struct magma_device_info* g_magma_devices = NULL;


// =============================================================================
// initialization

/***************************************************************************//**
    Initializes the MAGMA library.
    Caches information about available CUDA devices.

    Every magma_init call must be paired with a magma_finalize call.
    Only one thread needs to call magma_init and magma_finalize,
    but every thread may call it. If n threads call magma_init,
    the n-th call to magma_finalize will release resources.

    When renumbering CUDA devices, call cudaSetValidDevices before calling magma_init.
    When setting CUDA device flags, call hipSetDeviceFlags before calling magma_init.

    @retval MAGMA_SUCCESS
    @retval MAGMA_ERR_UNKNOWN
    @retval MAGMA_ERR_HOST_ALLOC

    @see magma_finalize

    @ingroup magma_init
*******************************************************************************/
extern "C" magma_int_t
magma_init()
{
    magma_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init == 0 ) {
            // query number of devices
            hipError_t err;
            g_magma_devices_cnt = 0;
            err = hipGetDeviceCount( &g_magma_devices_cnt );
            if ( err != 0 && err != hipErrorNoDevice ) {
                info = MAGMA_ERR_UNKNOWN;
                goto cleanup;
            }

            // allocate list of devices
            size_t size;
            size = max( 1, g_magma_devices_cnt ) * sizeof(struct magma_device_info);
            magma_malloc_cpu( (void**) &g_magma_devices, size );
            if ( g_magma_devices == NULL ) {
                info = MAGMA_ERR_HOST_ALLOC;
                goto cleanup;
            }
            memset( g_magma_devices, 0, size );

            // query each device
            for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
                hipDeviceProp_t prop;
                err = hipGetDeviceProperties( &prop, dev );
                if ( err != 0 ) {
                    info = MAGMA_ERR_UNKNOWN;
                }
                else {
                    g_magma_devices[dev].memory          = prop.totalGlobalMem;
                    g_magma_devices[dev].shmem_block     = prop.sharedMemPerBlock;
                    #ifdef MAGMA_HAVE_CUDA
                    g_magma_devices[dev].cuda_arch       = prop.major*100 + prop.minor*10;
                    g_magma_devices[dev].shmem_multiproc = prop.sharedMemPerMultiprocessor;
                    #elif defined(MAGMA_HAVE_HIP)
                    g_magma_devices[dev].cuda_arch       = prop.gcnArch;
                    g_magma_devices[dev].shmem_multiproc = prop.maxSharedMemoryPerMultiProcessor;
                    #endif

                    g_magma_devices[dev].multiproc_count = prop.multiProcessorCount;
                }
            }

            #ifndef MAGMA_NO_V1
                #ifdef HAVE_PTHREAD_KEY
                    // create thread-specific key
                    // currently, this is needed only for MAGMA v1 compatability
                    // see magma_init, magmablas(Set|Get)KernelStream, magmaGetQueue
                    info = pthread_key_create( &g_magma_queue_key, NULL );
                    if ( info != 0 ) {
                        info = MAGMA_ERR_UNKNOWN;
                        goto cleanup;
                    }
                #endif

                // ----- queues with NULL streams (for backwards compatability with MAGMA 1.x)
                // allocate array of queues with NULL stream
                size = max( 1, g_magma_devices_cnt ) * sizeof(magma_queue_t);
                magma_malloc_cpu( (void**) &g_null_queues, size );
                if ( g_null_queues == NULL ) {
                    info = MAGMA_ERR_HOST_ALLOC;
                    goto cleanup;
                }
                memset( g_null_queues, 0, size );
            #endif // MAGMA_NO_V1
        }
cleanup:
        g_init += 1;  // increment (init - finalize) count
    }
    g_mutex.unlock();

    return info;
}


/***************************************************************************//**
    Frees information used by the MAGMA library.
    @see magma_init

    @ingroup magma_init
*******************************************************************************/
extern "C" magma_int_t
magma_finalize()
{
    magma_int_t info = 0;

    g_mutex.lock();
    {
        if ( g_init <= 0 ) {
            info = MAGMA_ERR_NOT_INITIALIZED;
        }
        else {
            g_init -= 1;  // decrement (init - finalize) count
            if ( g_init == 0 ) {
                info = 0;

                if ( g_magma_devices != NULL ) {
                    magma_free_cpu( g_magma_devices );
                    g_magma_devices = NULL;
                }

                #ifndef MAGMA_NO_V1
                if ( g_null_queues != NULL ) {
                    for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
                        magma_queue_destroy( g_null_queues[dev] );
                        g_null_queues[dev] = NULL;
                    }
                    magma_free_cpu( g_null_queues );
                    g_null_queues = NULL;
                }

                #ifdef HAVE_PTHREAD_KEY
                    pthread_key_delete( g_magma_queue_key );
                #endif
                #endif // MAGMA_NO_V1

                #ifdef DEBUG_MEMORY
                magma_warn_leaks( g_pointers_dev, "device" );
                magma_warn_leaks( g_pointers_cpu, "CPU" );
                magma_warn_leaks( g_pointers_pin, "CPU pinned" );
                #endif
            }
        }
    }
    g_mutex.unlock();

    return info;
}


// =============================================================================
// testing and debugging support

#ifdef DEBUG_MEMORY
/***************************************************************************//**
    If DEBUG_MEMORY is defined at compile time, prints warnings when
    magma_finalize() is called for any GPU device, CPU, or CPU pinned
    allocations that were not freed.

    @param[in]
    pointers    Hash table mapping allocated pointers to size.

    @param[in]
    type        String describing type of pointers (GPU, CPU, etc.)

    @ingroup magma_testing
*******************************************************************************/
extern "C" void
magma_warn_leaks( const std::map< void*, size_t >& pointers, const char* type )
{
    if ( pointers.size() > 0 ) {
        fprintf( stderr, "Warning: MAGMA detected memory leak of %llu %s pointers:\n",
                 (long long unsigned) pointers.size(), type );
        std::map< void*, size_t >::const_iterator iter;
        for( iter = pointers.begin(); iter != pointers.end(); ++iter ) {
            fprintf( stderr, "    pointer %p, size %lu\n", iter->first, iter->second );
        }
    }
}
#endif


/***************************************************************************//**
    Print MAGMA version, CUDA version, LAPACK/BLAS library version,
    available GPU devices, number of threads, date, etc.
    Used in testing.
    @ingroup magma_testing
*******************************************************************************/
extern "C" void
magma_print_environment()
{
    magma_int_t major, minor, micro;
    magma_version( &major, &minor, &micro );

    printf( "%% MAGMA %lld.%lld.%lld %s %lld-bit magma_int_t, %lld-bit pointer.\n",
            (long long) major, (long long) minor, (long long) micro,
            MAGMA_VERSION_STAGE,
            (long long) (8*sizeof(magma_int_t)),
            (long long) (8*sizeof(void*)) );

/* CUDA */

#if defined(MAGMA_HAVE_CUDA)

    printf("%% Compiled with CUDA support for %.1f\n", MAGMA_CUDA_ARCH_MIN/100.);

    // CUDA, OpenCL, OpenMP, MKL, ACML versions all printed on same line
    int cuda_runtime=0, cuda_driver=0;
    hipError_t err;
    err = hipDriverGetVersion( &cuda_driver );
    check_error( err );
    err = hipRuntimeGetVersion( &cuda_runtime );
    if ( err != hipErrorNoDevice ) {
        check_error( err );
    }
    printf( "%% CUDA runtime %d, driver %d. ", cuda_runtime, cuda_driver );

#endif

/* HIP */

#if defined(MAGMA_HAVE_HIP)
    // TODO: add more specifics here

    int hip_runtime=0, hip_driver=0;
    hipError_t err;
    err = hipDriverGetVersion( &hip_driver );
    check_error( err );
    err = hipRuntimeGetVersion( &hip_runtime );
    if ( err != hipErrorNoDevice ) {
        check_error( err );
    }

    printf("%% HIP runtime %d, driver %d. ", hip_runtime, hip_driver );
#endif


/* OpenMP */

#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "OpenMP threads %d. ", omp_threads );
#else
    printf( "MAGMA not compiled with OpenMP. " );
#endif

#if defined(MAGMA_WITH_MKL)
    MKLVersion mkl_version;
    mkl_get_version( &mkl_version );
    printf( "MKL %d.%d.%d, MKL threads %d. ",
            mkl_version.MajorVersion,
            mkl_version.MinorVersion,
            mkl_version.UpdateVersion,
            mkl_get_max_threads() );
#endif

#if defined(MAGMA_WITH_ACML)
    // ACML 4 doesn't have acml_build parameter
    int acml_major, acml_minor, acml_patch, acml_build;
    acmlversion( &acml_major, &acml_minor, &acml_patch, &acml_build );
    printf( "ACML %d.%d.%d.%d ", acml_major, acml_minor, acml_patch, acml_build );
#endif

    printf( "\n" );

    // print devices
    int ndevices = 0;
    err = hipGetDeviceCount( &ndevices );
    if ( err != hipErrorNoDevice ) {
        check_error( err );
    }
    for( int dev = 0; dev < ndevices; ++dev ) {
        hipDeviceProp_t prop;
        err = hipGetDeviceProperties( &prop, dev );
        check_error( err );

        #ifdef MAGMA_HAVE_CUDA
        printf( "%% device %d: %s, %.1f MHz clock, %.1f MiB memory, capability %d.%d\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );

        int arch = prop.major*100 + prop.minor*10;
        if ( arch < MAGMA_CUDA_ARCH_MIN ) {
            printf("\n"
                   "==============================================================================\n"
                   "WARNING: MAGMA was compiled only for CUDA capability %.1f and higher;\n"
                   "device %d has only capability %.1f; some routines will not run correctly!\n"
                   "==============================================================================\n\n",
                   MAGMA_CUDA_ARCH_MIN/100., dev, arch/100. );
        }
        #endif

        #ifdef MAGMA_HAVE_HIP
        printf( "%% device %d: %s, %.1f MHz clock, %.1f MiB memory, gcn arch %d\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.gcnArch );
        #endif
    }

    MAGMA_UNUSED( err );
    time_t t = time( NULL );
    printf( "%% %s", ctime( &t ));
}

#if CUDA_VERSION >= 11000
#define magma_memoryType() type
#else
#define magma_memoryType() memoryType
#endif

/***************************************************************************//**
    For debugging purposes, determines whether a pointer points to CPU or GPU memory.

    On CUDA architecture 2.0 cards with unified addressing, CUDA can tell if
    it is a device pointer or pinned host pointer.
    For malloc'd host pointers, hipPointerGetAttributes returns error,
    implying it is a (non-pinned) host pointer.

    On older cards, this cannot determine if it is CPU or GPU memory.

    @param[in] A    pointer to test

    @return  1:  if A is a device pointer (definitely),
    @return  0:  if A is a host   pointer (definitely or inferred from error),
    @return -1:  if unknown.

    @ingroup magma_util
*******************************************************************************/
extern "C" magma_int_t
magma_is_devptr( const void* A )
{
    hipError_t err;
    hipDeviceProp_t prop;
    hipPointerAttribute_t attr;
    int dev;  // must be int
    err = hipGetDevice( &dev );
    if ( ! err ) {
        err = hipGetDeviceProperties( &prop, dev );

        #ifdef MAGMA_HAVE_CUDA
        if ( ! err && prop.unifiedAddressing ) {
        #elif defined(MAGMA_HAVE_HIP)
        // in HIP, assume all can.
        // There's no corresponding property, and examples show no need to check any properties
        if ( ! err ) {
        #endif

            // I think the hipPointerGetAttributes prototype is wrong, missing const (mgates)
            err = hipPointerGetAttributes( &attr, const_cast<void*>( A ));
            if ( ! err ) {
                // definitely know type
                #ifdef MAGMA_HAVE_CUDA
                  #if CUDA_VERSION >= 11000
                    return (attr.type == cudaMemoryTypeDevice);
                  #else
                    return (attr.memoryType == cudaMemoryTypeDevice);
                  #endif

                #elif defined(MAGMA_HAVE_HIP)
                return (attr.memoryType == hipMemoryTypeDevice);
                #endif
            }
            else if ( err == hipErrorInvalidValue ) {
                // clear error; see http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=529
                hipGetLastError();
                // infer as host pointer
                return 0;
            }
        }
    }
    // clear error
    hipGetLastError();
    // unknown, e.g., device doesn't support unified addressing
    return -1;
}


// =============================================================================
// device support

/***************************************************************************//**
    Returns CUDA architecture capability for the current device.
    This requires magma_init() to be called first to cache the information.
    Version is an integer xyz, where x is major, y is minor, and z is micro,
    the same as __CUDA_ARCH__. Thus for architecture 1.3.0 it returns 130.

    @return CUDA_ARCH for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" magma_int_t
magma_getdevice_arch()
{
    int dev;
    hipError_t err;
    err = hipGetDevice( &dev );
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}


/***************************************************************************//**
    Fills in devices array with the available devices.
    (This makes much more sense in OpenCL than in CUDA.)

    @param[out]
    devices     Array of dimension (size).
                On output, devices[0, ..., num_dev-1] contain device IDs.
                Entries >= num_dev are not touched.

    @param[in]
    size        Dimension of the array devices.

    @param[out]
    num_dev     Number of devices, limited to size.

    @ingroup magma_device
*******************************************************************************/
extern "C" void
magma_getdevices(
    magma_device_t* devices,
    magma_int_t  size,
    magma_int_t* num_dev )
{
    hipError_t err;
    int cnt;
    err = hipGetDeviceCount( &cnt );
    check_error( err );
    MAGMA_UNUSED( err );

    cnt = min( cnt, int(size) );
    for( int i = 0; i < cnt; ++i ) {
        devices[i] = i;
    }
    *num_dev = cnt;
}


/***************************************************************************//**
    Get the current device.

    @param[out]
    device      On output, device ID of the current device.
                Each thread has its own current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" void
magma_getdevice( magma_device_t* device )
{
    int dev;
    hipError_t err;
    err = hipGetDevice( &dev );
    *device = dev;
    check_error( err );
    MAGMA_UNUSED( err );
}


/***************************************************************************//**
    Set the current device.

    @param[in]
    device      Device ID to set as the current device.
                Each thread has its own current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" void
magma_setdevice( magma_device_t device )
{
    hipError_t err;
    err = hipSetDevice( int(device) );
    check_error( err );
    MAGMA_UNUSED( err );
}

/***************************************************************************//**
    Returns the multiprocessor count for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the multiprocessor count for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" magma_int_t
magma_getdevice_multiprocessor_count()
{
    int dev;
    hipError_t err;
    err = hipGetDevice( &dev );
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].multiproc_count;
}

/***************************************************************************//**
    Returns the maximum shared memory per block (in bytes) for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the maximum shared memory per block (in bytes) for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" size_t
magma_getdevice_shmem_block()
{
    int dev;
    hipError_t err;
    err = hipGetDevice( &dev );
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].shmem_block;
}


/***************************************************************************//**
    Returns the maximum shared memory multiprocessor (in bytes) for the current device.
    This requires magma_init() to be called first to cache the information.

    @return the maximum shared memory per multiprocessor (in bytes) for the current device.

    @ingroup magma_device
*******************************************************************************/
extern "C" size_t
magma_getdevice_shmem_multiprocessor()
{
    int dev;
    hipError_t err;
    err = hipGetDevice( &dev );
    check_error( err );
    MAGMA_UNUSED( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].shmem_multiproc;
}


/***************************************************************************//**
    @param[in]
    queue           Queue to query.

    @return         Amount of free memory in bytes available on the device
                    associated with the queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C" size_t
magma_mem_size( magma_queue_t queue )
{
    // CUDA would only need a device ID, but OpenCL requires a queue.
    size_t freeMem, totalMem;
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_setdevice( magma_queue_get_device( queue ));
    hipError_t err = hipMemGetInfo( &freeMem, &totalMem );
    check_error( err );
    MAGMA_UNUSED( err );
    magma_setdevice( orig_dev );
    return freeMem;
}


// =============================================================================
// queue support

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return Device ID associated with the MAGMA queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
magma_int_t
magma_queue_get_device( magma_queue_t queue )
{
    return queue->device();
}


#ifdef MAGMA_HAVE_CUDA
/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return CUDA stream associated with the MAGMA queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
hipStream_t
magma_queue_get_cuda_stream( magma_queue_t queue )
{
    return queue->hip_stream();
}


/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return cuBLAS handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/

extern "C"
hipblasHandle_t
magma_queue_get_cublas_handle( magma_queue_t queue )
{
    return queue->hipblas_handle();
}

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return cuSparse handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
hipsparseHandle_t
magma_queue_get_cusparse_handle( magma_queue_t queue )
{
    return queue->cusparse_handle();
}

#elif defined(MAGMA_HAVE_HIP)

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return HIP stream associated with the MAGMA queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
hipStream_t
magma_queue_get_hip_stream( magma_queue_t queue )
{
    return queue->hip_stream();
}


/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return hipBLAS handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/

extern "C"
hipblasHandle_t
magma_queue_get_hipblas_handle( magma_queue_t queue )
{
    return queue->hipblas_handle();
}

/***************************************************************************//**
    @param[in]
    queue       Queue to query.

    @return hipSparse handle associated with the MAGMA queue.
            MAGMA assumes the handle's stream will not be modified.

    @ingroup magma_queue
*******************************************************************************/
extern "C"
hipsparseHandle_t
magma_queue_get_hipsparse_handle( magma_queue_t queue )
{
    return queue->hipsparse_handle();
}



#endif



/***************************************************************************//**
    @fn magma_queue_create( device, queue_ptr )

    magma_queue_create( device, queue_ptr ) is the preferred alias to this
    function.

    Creates a new MAGMA queue, with associated CUDA stream, cuBLAS handle,
    and cuSparse handle.

    This is the MAGMA v2 version which takes a device ID.

    @param[in]
    device          Device to create queue on.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup magma_queue
*******************************************************************************/
extern "C" void
magma_queue_create_internal(
    magma_device_t device, magma_queue_t* queue_ptr,
    const char* func, const char* file, int line )
{
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;

#if defined(MAGMA_HAVE_CUDA)
    queue->hipblas__   = NULL;
    queue->cusparse__ = NULL;
#elif defined(MAGMA_HAVE_HIP)
    queue->hipblas__  = NULL;
    queue->hipsparse__ = NULL;
#endif
    queue->maxbatch__ = MAX_BATCHCOUNT;

    magma_setdevice( device );

    hipError_t err;
    err = hipStreamCreate( &queue->stream__ );
    check_xerror( err, func, file, line );
    queue->own__ |= own_stream;

#if defined(MAGMA_HAVE_CUDA)
    hipblasStatus_t stat;
    stat = hipblasCreate( &queue->hipblas__ );
    check_xerror( stat, func, file, line );
    queue->own__ |= own_cublas;
    stat = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    hipsparseStatus_t stat2;
    stat2 = hipsparseCreate( &queue->cusparse__ );
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_cusparse;
    stat2 = hipsparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );
#elif defined(MAGMA_HAVE_HIP)

    hipblasStatus_t stat;
    stat = hipblasCreate( &queue->hipblas__ );
    check_xerror( stat, func, file, line );
    queue->own__ |= own_hipblas;
    stat = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    hipsparseStatus_t stat2;
    stat2 = hipsparseCreate( &queue->hipsparse__ );
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_hipsparse;
    stat2 = hipsparseSetStream( queue->hipsparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

#endif

    MAGMA_UNUSED( err );
    MAGMA_UNUSED( stat );
    MAGMA_UNUSED( stat2 );
}


/***************************************************************************//**
    @fn magma_queue_create_from_cuda( device, hip_stream, hipblas_handle, cusparse_handle, queue_ptr )

    Warning: non-portable outside of CUDA. Use with discretion.

    Creates a new MAGMA queue, using the given CUDA stream, cuBLAS handle, and
    cuSparse handle. The caller retains ownership of the given stream and
    handles, so must free them after destroying the queue;
    see magma_queue_destroy().

    MAGMA sets the stream on the cuBLAS and cuSparse handles, and assumes
    it will not be changed while MAGMA is running.

    @param[in]
    device          Device to create queue on.

    @param[in]
    hip_stream     CUDA stream to use, even if NULL (the so-called default stream).

    @param[in]
    hipblas_handle   cuBLAS handle to use. If NULL, a new handle is created.

    @param[in]
    cusparse_handle cuSparse handle to use. If NULL, a new handle is created.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup magma_queue
*******************************************************************************/
#ifdef MAGMA_HAVE_CUDA
extern "C" void
magma_queue_create_from_cuda_internal(
    magma_device_t   device,
    hipStream_t     hip_stream,
    hipblasHandle_t   hipblas_handle,
    hipsparseHandle_t cusparse_handle,
    magma_queue_t*   queue_ptr,
    const char* func, const char* file, int line )
{
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->hipblas__   = NULL;
    queue->cusparse__ = NULL;
    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;
    queue->maxbatch__ = MAX_BATCHCOUNT;

    magma_setdevice( device );

    // stream can be NULL
    queue->stream__ = hip_stream;

    // allocate hipblas handle if given as NULL
    hipblasStatus_t stat;
    if ( hipblas_handle == NULL ) {
        stat  = hipblasCreate( &hipblas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cublas;
    }
    queue->hipblas__ = hipblas_handle;
    stat  = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    // allocate cusparse handle if given as NULL
    hipsparseStatus_t stat2;
    if ( cusparse_handle == NULL ) {
        stat2 = hipsparseCreate( &cusparse_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cusparse;
    }
    queue->cusparse__ = cusparse_handle;
    stat2 = hipsparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

    MAGMA_UNUSED( stat );
    MAGMA_UNUSED( stat2 );

}
#endif


/***************************************************************************//**
    @fn magma_queue_create_from_hip( device, hip_stream, hipblas_handle, hipsparse_handle, queue_ptr )

    Warning: non-portable outside of CUDA. Use with discretion.

    Creates a new MAGMA queue, using the given CUDA stream, cuBLAS handle, and
    cuSparse handle. The caller retains ownership of the given stream and
    handles, so must free them after destroying the queue;
    see magma_queue_destroy().

    MAGMA sets the stream on the cuBLAS and cuSparse handles, and assumes
    it will not be changed while MAGMA is running.

    @param[in]
    device          Device to create queue on.

    @param[in]
    hip_stream     CUDA stream to use, even if NULL (the so-called default stream).

    @param[in]
    hipblas_handle   cuBLAS handle to use. If NULL, a new handle is created.

    @param[in]
    cusparse_handle cuSparse handle to use. If NULL, a new handle is created.

    @param[out]
    queue_ptr       On output, the newly created queue.

    @ingroup magma_queue
*******************************************************************************/
#ifdef MAGMA_HAVE_HIP
extern "C" void
magma_queue_create_from_hip_internal(
    magma_device_t    device,
    hipStream_t       hip_stream,
    hipblasHandle_t   hipblas_handle,
    hipsparseHandle_t hipsparse_handle,
    magma_queue_t*    queue_ptr,
    const char* func, const char* file, int line )
{
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;

    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;

    queue->ptrArray__ = NULL;
    queue->dAarray__  = NULL;
    queue->dBarray__  = NULL;
    queue->dCarray__  = NULL;

    queue->hipblas__  = NULL;
    queue->hipsparse__= NULL;
    queue->maxbatch__ = MAX_BATCHCOUNT;

    magma_setdevice( device );

    // stream can be NULL
    queue->stream__ = hip_stream;

    // allocate hipblas handle if given as NULL
    hipblasStatus_t stat;
    if ( hipblas_handle == NULL ) {
        stat  = hipblasCreate( &hipblas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_hipblas;
    }
    queue->hipblas__ = hipblas_handle;
    stat  = hipblasSetStream( queue->hipblas__, queue->stream__ );
    check_xerror( stat, func, file, line );

    // allocate cusparse handle if given as NULL
    hipsparseStatus_t stat2;
    if ( hipsparse_handle == NULL ) {
        stat2 = hipsparseCreate( &hipsparse_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_hipsparse;
    }
    queue->hipsparse__ = hipsparse_handle;
    stat2 = hipsparseSetStream( queue->hipsparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );

    MAGMA_UNUSED( stat );
    MAGMA_UNUSED( stat2 );
}
#endif



/***************************************************************************//**
    @fn magma_queue_destroy( queue )

    Destroys a queue, freeing its resources.

    If the queue was created with magma_queue_create_from_cuda(), the CUDA
    stream, cuBLAS handle, and cuSparse handle given there are NOT freed -- the
    caller retains ownership. However, if MAGMA allocated the handles, MAGMA
    will free them here.

    @param[in]
    queue           Queue to destroy.

    @ingroup magma_queue
*******************************************************************************/
extern "C" void
magma_queue_destroy_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    if ( queue != NULL ) {
    #if defined(MAGMA_HAVE_CUDA)
        if ( queue->hipblas__ != NULL && (queue->own__ & own_cublas)) {
            hipblasStatus_t stat = hipblasDestroy( queue->hipblas__ );
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
        if ( queue->cusparse__ != NULL && (queue->own__ & own_cusparse)) {
            hipsparseStatus_t stat = hipsparseDestroy( queue->cusparse__ );
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
    #elif defined(MAGMA_HAVE_HIP)

        if ( queue->hipblas__ != NULL && (queue->own__ & own_hipblas)) {
            hipblasStatus_t stat = hipblasDestroy( queue->hipblas__ );
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
        if ( queue->hipsparse__ != NULL && (queue->own__ & own_hipsparse)) {
            hipsparseStatus_t stat = hipsparseDestroy( queue->hipsparse__ );
            check_xerror( stat, func, file, line );
            MAGMA_UNUSED( stat );
        }
    #endif
        if ( queue->stream__ != NULL && (queue->own__ & own_stream)) {
            hipError_t err = hipStreamDestroy( queue->stream__ );
            check_xerror( err, func, file, line );
            MAGMA_UNUSED( err );
        }

        if( queue->ptrArray__ != NULL ) magma_free( queue->ptrArray__ );

        queue->own__      = own_none;
        queue->device__   = -1;
        queue->stream__   = NULL;
        queue->ptrArray__ = NULL;
        queue->dAarray__  = NULL;
        queue->dBarray__  = NULL;
        queue->dCarray__  = NULL;

    #if defined(MAGMA_HAVE_CUDA)
        queue->hipblas__   = NULL;
        queue->cusparse__ = NULL;
    #elif defined(MAGMA_HAVE_HIP)
        queue->hipblas__  = NULL;
        queue->hipsparse__= NULL;
    #endif

        magma_free_cpu( queue );
    }
}


/***************************************************************************//**
    @fn magma_queue_sync( queue )

    Synchronizes with a queue. The CPU blocks until all operations on the queue
    are finished.

    @param[in]
    queue           Queue to synchronize.

    @ingroup magma_queue
*******************************************************************************/
extern "C" void
magma_queue_sync_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    hipError_t err;
    if ( queue != NULL ) {
        err = hipStreamSynchronize( queue->hip_stream() );
    }
    else {
        err = hipStreamSynchronize( NULL );
    }
    check_xerror( err, func, file, line );
    MAGMA_UNUSED( err );
}


// =============================================================================
// event support

/***************************************************************************//**
    Creates a GPU event.

    @param[in]
    event           On output, the newly created event.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_event_create( magma_event_t* event )
{
    hipError_t err;
    err = hipEventCreate( event );
    check_error( err );
    MAGMA_UNUSED( err );
}


/***************************************************************************//**
    Creates a GPU event, without timing support. May improve performance

    @param[in]
    event           On output, the newly created event.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_event_create_untimed( magma_event_t* event )
{
    hipError_t err;
    err = hipEventCreateWithFlags( event, hipEventDisableTiming );
    check_error( err );
    MAGMA_UNUSED( err );
}



/***************************************************************************//*
    Destroys a GPU event, freeing its resources.

    @param[in]
    event           Event to destroy.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_event_destroy( magma_event_t event )
{
    if ( event != NULL ) {
        hipError_t err;
        err = hipEventDestroy( event );
        check_error( err );
        MAGMA_UNUSED( err );
    }
}


/***************************************************************************//**
    Records an event into the queue's execution stream.
    The event will trigger when all previous operations on this queue finish.

    @param[in]
    event           Event to record.

    @param[in]
    queue           Queue to execute in.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_event_record( magma_event_t event, magma_queue_t queue )
{
    hipError_t err;
    err = hipEventRecord( event, queue->hip_stream() );
    check_error( err );
    MAGMA_UNUSED( err );
}


/***************************************************************************//**
    Synchronizes with an event. The CPU blocks until the event triggers.

    @param[in]
    event           Event to synchronize with.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_event_sync( magma_event_t event )
{
    hipError_t err;
    err = hipEventSynchronize( event );
    check_error( err );
    MAGMA_UNUSED( err );
}


/***************************************************************************//**
    Synchronizes a queue with an event. The queue blocks until the event
    triggers. The CPU does not block.

    @param[in]
    event           Event to synchronize with.

    @param[in]
    queue           Queue to synchronize.

    @ingroup magma_event
*******************************************************************************/
extern "C" void
magma_queue_wait_event( magma_queue_t queue, magma_event_t event )
{
    hipError_t err;
    err = hipStreamWaitEvent( queue->hip_stream(), event, 0 );
    check_error( err );
    MAGMA_UNUSED( err );
}

#endif // MAGMA_HAVE_CUDA or MAGMA_HAVE_HIP
