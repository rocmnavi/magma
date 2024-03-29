!
!   -- MAGMA (version 2.7.2) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date August 2023
!

module magma_param

    implicit none

    ! could use STORAGE_SIZE in Fortran 2008
    integer, parameter :: sizeof_complex_16 = 16
    integer, parameter :: sizeof_complex    = 8
    integer, parameter :: sizeof_double     = 8
    integer, parameter :: sizeof_real       = 4
    
#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
    integer, parameter :: sizeof_integer    = 8
#else
    integer, parameter :: sizeof_integer    = 4
#endif

end module magma_param
