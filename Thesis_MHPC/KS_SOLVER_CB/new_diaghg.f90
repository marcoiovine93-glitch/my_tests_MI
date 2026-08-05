!
#define ZERO ( 0.D0, 0.D0 )
#define ONE  ( 1.D0, 0.D0 )

SUBROUTINE laxlib_cdiaghg_gpu( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
  !----------------------------------------------------------------------------
  !!
  !! Called by diaghg interface.
  !! Calculates eigenvalues and eigenvectors of the generalized problem
  !! Solve Hv = eSv, with H symmetric matrix, S overlap matrix.
  !! complex matrices version.
  !! On output both matrix are unchanged.
  !!
  !! GPU VERSION.
  !
#if defined(_OPENMP)
  USE omp_lib
#endif
  !
#if defined(__CUDA)
  USE cudafor
  !
  USE cusolverdn
  USE laxlib_cusolver_handles, ONLY : cusolver_handle, cusolver_initialized, laxlib_cuda_stream, & 
                                      cusolver_thread 
#endif
  !
  USE laxlib_parallel_include
  !
  ! NB: the flag below can be used to decouple LAXlib from devXlib.
  !     This will make devXlib an optional dependency of LAXlib when
  !     the library will be decoupled from QuantumESPRESSO.
#if defined(__USE_GLOBAL_BUFFER) && defined(__CUDA)
  USE device_fbuff_m,        ONLY : dev=>dev_buf, pin=>pin_buf
#define VARTYPE POINTER
#else
#define VARTYPE ALLOCATABLE
#endif
  !
  IMPLICIT NONE
  include 'laxlib_kinds.fh'
  !
  INTEGER, INTENT(IN) :: n
  !! dimension of the matrix to be diagonalized
  INTEGER, INTENT(IN) :: m
  !! number of eigenstates to be calculated
  INTEGER, INTENT(IN) :: ldh
  !! leading dimension of h, as declared in the calling pgm unit
  COMPLEX(DP), INTENT(INOUT) :: h_d(ldh,n)
  !! matrix to be diagonalized, allocated on the GPU
  COMPLEX(DP), INTENT(INOUT) :: s_d(ldh,n)
  !! overlap matrix, allocated on the GPU
  REAL(DP), INTENT(OUT) :: e_d(n)
  !! eigenvalues, , allocated on the GPU
  COMPLEX(DP),  INTENT(OUT) :: v_d(ldh,n)
  !! eigenvectors (column-wise), , allocated on the GPU
  INTEGER,  INTENT(IN)  :: me_bgrp
  !! index of the processor within a band group
  INTEGER,  INTENT(IN)  :: root_bgrp
  !! index of the root processor within a band group
  INTEGER,  INTENT(IN)  :: intra_bgrp_comm
  !! intra band group communicator
  !
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: h_d, s_d, e_d, v_d
#endif
  !
  INTEGER              :: lwork, info
  !
  REAL(DP)             :: abstol
  INTEGER, ALLOCATABLE :: ifail(:)
  INTEGER, VARTYPE     :: iwork(:)
  REAL(DP), VARTYPE    :: rwork(:)
  COMPLEX(DP), VARTYPE :: work(:)
  !
  COMPLEX(DP), VARTYPE :: v_h(:,:)
  REAL(DP), VARTYPE    :: e_h(:)
#if (! defined(__USE_GLOBAL_BUFFER)) && defined(__CUDA)
  ATTRIBUTES( PINNED ) :: work, iwork, rwork, v_h, e_h
#endif
  !
  INTEGER              :: lwork_d, lrwork_d, liwork, lrwork
  REAL(DP), VARTYPE    :: rwork_d(:)
  COMPLEX(DP), VARTYPE :: work_d(:)
  ! various work space
  !
  ! Temp arrays to save H and S.
  REAL(DP), VARTYPE    :: h_diag_d(:), s_diag_d(:)
#if defined(__CUDA)
  ATTRIBUTES( DEVICE ) :: work_d, rwork_d, h_diag_d, s_diag_d
  INTEGER                      :: devInfo_d, h_meig
  ATTRIBUTES( DEVICE )         :: devInfo_d
  TYPE(cusolverDnHandle), SAVE :: cuSolverHandle
  LOGICAL, SAVE                :: cuSolverInitialized = .FALSE.
  !
  COMPLEX(DP), VARTYPE   :: h_bkp_d(:,:), s_bkp_d(:,:)
  ATTRIBUTES( DEVICE )   :: h_bkp_d, s_bkp_d
#endif
  INTEGER :: i, j
#undef VARTYPE
  !
  !
  !
  !
  CALL start_clock_gpu( 'cdiaghg' )
  !
  ! ... only the first processor diagonalizes the matrix
  !
  IF ( me_bgrp == root_bgrp ) THEN
      !
      ! Keeping compatibility for both CUSolver and CustomEigensolver, CUSolver below
      !
#if defined(__CUDA)

#if ! defined(__USE_GLOBAL_BUFFER)
      ALLOCATE(h_bkp_d(n,n), s_bkp_d(n,n), STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d or s_bkp_d ', ABS( info ) )
#else
      CALL dev%lock_buffer( h_bkp_d,  (/ n, n /), info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d ', ABS( info ) )
      CALL dev%lock_buffer( s_bkp_d,  (/ n, n /), info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate s_bkp_d ', ABS( info ) )
#endif
      !
!$cuf kernel do(2) <<<*,*,0,laxlib_cuda_stream>>>
      DO j=1,n
         DO i=1,n
            h_bkp_d(i,j) = h_d(i,j)
            s_bkp_d(i,j) = s_d(i,j)
         ENDDO
      ENDDO
      !
#if defined(_OPENMP)
      IF (omp_get_num_threads() > 1) CALL lax_error__( ' cdiaghg_gpu ', 'cdiaghg_gpu is not thread-safe',  ABS( info ) )
#endif
      IF ( .NOT. cusolver_initialized(cusolver_thread) ) THEN
         info = cusolverDnCreate(cusolver_handle(cusolver_thread))
         IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnCreate',  ABS( info ) )
         cusolver_initialized(cusolver_thread) = .TRUE.
         info = cusolverDnSetStream(cusolver_handle(cusolver_thread), laxlib_cuda_stream )
         IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnSetStream',  ABS( info ) )   
      ENDIF
      !
      cuSolverHandle = cusolver_handle(cusolver_thread)
      info = cusolverDnZhegvdx_bufferSize(cuSolverHandle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, &
                                               n, h_d, ldh, s_d, ldh, 0.D0, 0.D0, 1, m, h_meig, e_d, lwork_d)
      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZhegvdx_bufferSize failed ', ABS( info ) )
      !
#if ! defined(__USE_GLOBAL_BUFFER)
      ALLOCATE(work_d(1*lwork_d), STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
#else
      CALL dev%lock_buffer( work_d,  lwork_d, info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
#endif
      !
      info = cusolverDnZhegvdx(cuSolverHandle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, &
                                  n, h_d, ldh, s_d, ldh, 0.D0, 0.D0, 1, m, h_meig, e_d, work_d, lwork, devInfo_d)
      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZhegvdx failed ', ABS( info ) )
!$cuf kernel do(2) <<<*,*,0,laxlib_cuda_stream>>>
      DO j=1,n
         DO i=1,n
            IF(j <= m) v_d(i,j) = h_d(i,j)
            h_d(i,j) = h_bkp_d(i,j)
            s_d(i,j) = s_bkp_d(i,j)
         ENDDO
      ENDDO
      !
      !
      ! Do not destroy the handle to save the (re)creation time on each call.
      !
      !info = cusolverDnDestroy(cuSolverHandle)
      !IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnDestroy failed ', ABS( info ) )
      !
#if ! defined(__USE_GLOBAL_BUFFER)
      DEALLOCATE(work_d)
      DEALLOCATE(h_bkp_d, s_bkp_d)
#else
      CALL dev%release_buffer( work_d,  info )
      CALL dev%release_buffer( h_bkp_d, info )
      CALL dev%release_buffer( s_bkp_d, info )
#endif
      !
      ! Keeping compatibility for both CUSolver and CustomEigensolver, CustomEigensolver below
      !
#else
     CALL lax_error__( 'cdiaghg', 'Called GPU eigensolver without GPU support', 1 )
#endif
     !
  END IF
  !
  ! ... broadcast eigenvectors and eigenvalues to all other processors
  !
#if defined __MPI
#if defined __GPU_MPI
  info = cudaDeviceSynchronize()
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error synchronizing device (first)', ABS( info ) )
  CALL MPI_BCAST( e_d, n, MPI_DOUBLE_PRECISION, root_bgrp, intra_bgrp_comm, info )
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error broadcasting array e_d', ABS( info ) )
  CALL MPI_BCAST( v_d, ldh*m, MPI_DOUBLE_COMPLEX, root_bgrp, intra_bgrp_comm, info )
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error broadcasting array v_d', ABS( info ) )
  info = cudaDeviceSynchronize() ! this is probably redundant...
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error synchronizing device (second)', ABS( info ) )
#else
  ALLOCATE(e_h(n), v_h(ldh,m))
  e_h(1:n) = e_d(1:n)
  v_h(1:ldh, 1:m) = v_d(1:ldh, 1:m)
  CALL MPI_BCAST( e_h, n, MPI_DOUBLE_PRECISION, root_bgrp, intra_bgrp_comm, info )
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error broadcasting array e_d', ABS( info ) )
  CALL MPI_BCAST( v_h, ldh*m, MPI_DOUBLE_COMPLEX, root_bgrp, intra_bgrp_comm, info )
  IF ( info /= 0 ) &
        CALL lax_error__( 'cdiaghg', 'error broadcasting array v_d', ABS( info ) )
  e_d(1:n) = e_h(1:n)
  v_d(1:ldh, 1:m) = v_h(1:ldh, 1:m)
  DEALLOCATE(e_h, v_h)
#endif
#endif
  !
  CALL stop_clock_gpu( 'cdiaghg' )
  !
  RETURN
  !
END SUBROUTINE laxlib_cdiaghg_gpu
