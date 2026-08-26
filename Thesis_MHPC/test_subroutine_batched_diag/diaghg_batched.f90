SUBROUTINE laxlib_cdiaghg_gpu_batched( n, m, h_d, s_d, ldh, e_d, v_d, n_k, me_bgrp, root_bgrp, intra_bgrp_comm)
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
  USE ieee_arithmetic, ONLY: ieee_is_nan, ieee_is_finite !!! JUST FOR DEBUGGING!
  USE cublas !! M.Iovine - we need to explicitly include the use clublas
  !
  USE openacc, only: c_devptr !! M.Iovine - we include c_devptr
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
  !!!! M.Iovine - nbase is set equal to the maximum value among the 
  !!!! k-pints (threads)
  INTEGER, INTENT(IN) :: n
  !! dimension of the matrix to be diagonalized
  !!!! number of desired root eigenstates for each k point is the same
  !!!! for each k-point!
  INTEGER, INTENT(IN) :: m
  !! number of eigenstates to be calculated
  !!!! M.Iovine - Batched change:
  INTEGER, INTENT(IN) :: n_k  
  !!!!
  INTEGER, INTENT(IN) :: ldh
  !! leading dimension of h, as declared in the calling pgm unit
  COMPLEX(DP), INTENT(INOUT) :: h_d(ldh,n,n_k) !! M.Iovine - added a dimension -It is still a static array
  !! matrix to be diagonalized, allocated on the GPU
  COMPLEX(DP), INTENT(INOUT) :: s_d(ldh,n,n_k) !! M.Iovine - added a dimension -It is still a static array
  !! overlap matrix, allocated on the GPU
  !!!! M.Iovine - Batched change: the array was made 3D
  REAL(DP), INTENT(OUT) :: e_d(n,n_k)
  !!!!
  !! eigenvalues, , allocated on the GPU
  COMPLEX(DP),  INTENT(OUT) :: v_d(ldh,n,n_k) !!!! M.Iovine - v_d becomes a 3D array! It is still a static array
  !! eigenvectors (column-wise), , allocated on the GPU
  INTEGER,  INTENT(IN)  :: me_bgrp
  !! index of the processor within a band group
  INTEGER,  INTENT(IN)  :: root_bgrp
  !! index of the root processor within a band group
  INTEGER,  INTENT(IN)  :: intra_bgrp_comm
  !! intra band group communicator
  COMPLEX(DP), ALLOCATABLE :: h_sym_chk(:,:) !!M.Iovine - lines added for debugging
  REAL(DP) :: max_asym !!M.Iovine - lines added for debugging
  INTEGER :: ii_sym, jj_sym !!M.Iovine - lines added for debugging
  INTEGER, ALLOCATABLE :: dinfo_host(:)!! Added for debugging
  !
#if defined(__CUDA)
    ATTRIBUTES(DEVICE) :: h_d, s_d, e_d, v_d
#endif
  !
!!!! DEBUGGING LINES!1
#if defined(__CUDA)
    COMPLEX(DP), ALLOCATABLE :: nan_chk(:,:)
    LOGICAL, ALLOCATABLE :: nan_mask_r(:,:), nan_mask_i(:,:)
    LOGICAL :: has_nan_dbg
    REAL(DP), ALLOCATABLE :: nan_echk(:)
    LOGICAL, ALLOCATABLE :: nan_emask(:)
    LOGICAL, ALLOCATABLE :: inf_mask_r(:,:), inf_mask_i(:,:)
    LOGICAL :: has_inf_dbg
#endif
  
  
  INTEGER              :: lwork, info
  !
  REAL(DP)             :: abstol
  INTEGER, ALLOCATABLE :: ifail(:)
  INTEGER, VARTYPE     :: iwork(:)
  REAL(DP), VARTYPE    :: rwork(:)
  COMPLEX(DP), VARTYPE :: work(:)
  !
  COMPLEX(DP), VARTYPE :: v_h(:,:,:) !!!! M.Iovine : added a dimension to the array
  REAL(DP), VARTYPE    :: e_h(:,:)
#if (! defined(__USE_GLOBAL_BUFFER)) && defined(__CUDA)
  ATTRIBUTES( PINNED ) :: work, iwork, rwork, v_h, e_h
#endif
  !
  INTEGER              :: lwork_d, lrwork_d, liwork, lrwork
  REAL(DP), VARTYPE    :: rwork_d(:)
  COMPLEX(DP), VARTYPE :: work_d(:)
  !!!! M.Iovine - We add a line to define the info array corresponding to the Batched routine of the Cusolver NVIDIA library:
  INTEGER, ALLOCATABLE :: d_info(:)
  !!!! M.Iovine - We define host arrays of c_devpr pointers:
  TYPE(c_devptr), ALLOCATABLE :: arr_of_ptr_s(:)
  TYPE(c_devptr), ALLOCATABLE :: arr_of_ptr_h(:)
  !!!! M.Iovine - We define device arrays of c_devpr pointers:
  TYPE(c_devptr), ALLOCATABLE, DEVICE :: arr_of_ptr_h_d(:), arr_of_ptr_s_d(:)
  COMPLEX(DP) :: alpha=(1.D0, 0.D0) !! M.Iovine - we define the alpha coefficient for the triangular cublas linear solver 
  INTEGER :: b, ik,ind_min !! M.Iovine - indices for eigenvectors ordering 
  COMPLEX(DP) :: minim !! M.Iovine - min. value for padding
  COMPLEX(DP) :: min_temp !! M.Iovine - temp variable for swapping of arrays for ordering eigenvalues and eigenvect
  ! various work space
  !
  ! Temp arrays to save H and S.
  REAL(DP), VARTYPE    :: h_diag_d(:), s_diag_d(:)
#if defined(__CUDA)
  ATTRIBUTES( DEVICE ) :: work_d, rwork_d, h_diag_d, s_diag_d, d_info, arr_of_ptr_s_d, arr_of_ptr_h_d !!!! M.Iovine - added d_info and arr_of_ptr to the device attributes
  INTEGER                      :: devInfo_d, h_meig
  ATTRIBUTES( DEVICE )         :: devInfo_d
  !TYPE(cusolverDnHandle), SAVE :: cuSolverHandle !M.Iovine - commented line
  TYPE(cusolverDnHandle), SAVE :: cuSolverHandle_batched !! M.Iovine - we define a proper handle for the batched cdiaghg_gpu
  LOGICAL, SAVE                :: cuSolverInitialized = .FALSE.
  !
  !! Device arrays to save the old values of the Hamiltonian and Overlap matrices!!
  COMPLEX(DP), VARTYPE   :: h_bkp_d(:,:,:), s_bkp_d(:,:,:) !!!! M.Iovine : The array to store the old values of the Hamiltonian before appying the 
  ATTRIBUTES( DEVICE )   :: h_bkp_d, s_bkp_d

  !!!! M.Iovine - We instantiate a cusolverDnSyevjInfo variable that is needed for the Batched routine provided by NVIDIA:
  TYPE(cusolverDnSyevjInfo) :: syevj_params

  !! M.Iovine - we define the variables for the cublas handle:
  INTEGER(kind=cuda_stream_kind) :: mycudaStream
  type(cublasHandle) :: cublas_handle
  INTEGER :: istat_cublas

#endif
  INTEGER :: i, j, k !!!! M.IOvine - added index k for the third dimension of the arrays
  mycudaStream = laxlib_cuda_stream
  istat_cublas = cublasCreate(cublas_handle) !!! M.Iovine - introduced cublas handle
  istat_cublas = cublasSetStream(cublas_handle, mycudaStream) !!!M.IOvine - COMMENTED TO DEBUGG!!
  IF (istat_cublas /= 0) CALL lax_error__( ' cdiaghg_gpu ', 'cublasSetStream', ABS(istat_cublas) ) 
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
      print *, 'DEBUG: n =', n, ' m =', m, ' ldh =', ldh, ' n_k =', n_k
      !
      ! Keeping compatibility for both CUSolver and CustomEigensolver, CUSolver below
      !
#if defined(__CUDA)

#if ! defined(__USE_GLOBAL_BUFFER)
      ALLOCATE(h_bkp_d(n,n,n_k), s_bkp_d(n,n,n_k), STAT = info) !!!! M.Iovine : h_bkp_d is a 3 dimensional array now!!
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d or s_bkp_d ', ABS( info ) )
      !! M.Iovine - We allocate the array d_info necessary for the cusolver diagonalization:
      ALLOCATE(d_info(n_k),STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate d_info ', ABS( info ) )
      !! M.Iovine - we allocate the host arrays of pointers for the Cholesky factorization:
      ALLOCATE(arr_of_ptr_s(n_k),STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate arr_of_ptr_s ', ABS( info ) )
      ALLOCATE(arr_of_ptr_h(n_k),STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate arr_of_ptr_h ', ABS( info ) )
      !! M.Iovine - we allocate the device arrays of pointers for the Cholesky factorization:
      ALLOCATE(arr_of_ptr_s_d(n_k),STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate arr_of_ptr_s ', ABS( info ) )
      ALLOCATE(arr_of_ptr_h_d(n_k),STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate arr_of_ptr_h ', ABS( info ) )

#else
      CALL dev%lock_buffer( h_bkp_d,  (/ n, n /), info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d ', ABS( info ) )
      CALL dev%lock_buffer( s_bkp_d,  (/ n, n /), info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate s_bkp_d ', ABS( info ) )
#endif
      !
!$cuf kernel do(3) <<<*,*,0,laxlib_cuda_stream>>> !!!! M.Iovine - Modified the loops nested from 2 to 3 --> so we changed kernel do(2) to kenel do(3) :
      DO k=1,n_k
         DO j=1,n
            DO i=1,n
                h_bkp_d(i,j,k) = h_d(i,j,k)
                s_bkp_d(i,j,k) = s_d(i,j,k)
            ENDDO
         ENDDO
      ENDDO
      !i
!!!!DEBUG LINES:
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[1] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))
!!!!

#if defined(_OPENMP)
      IF (omp_get_num_threads() > 1) CALL lax_error__( ' cdiaghg_gpu ', 'cdiaghg_gpu is not thread-safe',  ABS( info ) )
#endif
      !IF ( .NOT. cusolver_initialized(cusolver_thread) ) THEN
       ! info = cusolverDnCreate(cusolver_handle(cusolver_thread))
        !IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnCreate',  ABS( info ) )
        !cusolver_initialized(cusolver_thread) = .TRUE.
        !info = cusolverDnSetStream(cusolver_handle(cusolver_thread), laxlib_cuda_stream )
        !IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnSetStream',  ABS( info ) )   
      !ENDIF
      IF( .NOT. cuSolverInitialized ) THEN   !!! M.Iovine - we introduce a new Handle to avoid leaving changes to the next calls done
                                                    !through the kernel calls inside the iterative loop!!
        info = cusolverDnCreate(cuSolverHandle_batched)
        IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu_batched ', 'cusolverDnCreate',  ABS( info ) )
         cuSolverInitialized = .TRUE.
         info = cusolverDnSetStream(cuSolverHandle_batched, laxlib_cuda_stream) 
         IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu_batched ', 'cusolverDnSetStream',  ABS( info ) )
      ENDIF 


    !!!! M.Iovine - Cholesky factorization of the s overlap matrix:
    !! We assign to each element of arr_of_ptr the device pointer to each 2D slice in s_d :
    do k = 1, n_k
        arr_of_ptr_s(k) = c_devloc(s_d(:,:,k))
    end do
    
    !!! M.Iovine - we copy the c_devptr in the host array arr_of_ptr_s to the device array arr_of_ptr_s_d
    arr_of_ptr_s_d = arr_of_ptr_s
    
    
    !!!DEBUGG :
    istat_cublas = cudaGetLastError()
IF (istat_cublas /= 0) THEN
   print *, 'STICKY CUDA ERROR before ZpotrfBatched: ', &
             cudaGetErrorString(istat_cublas)
END IF
    !!!!

     
    !cuSolverHandle = cusolver_handle(cusolver_thread) !!M.Iovine - this line must before any cuSolver routine kernel call!
    info = cusolverDnZpotrfBatched(cuSolverHandle_batched, CUBLAS_FILL_MODE_LOWER, n, arr_of_ptr_s_d, ldh, d_info(1), n_k)
    IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnZpotrfBatched',  ABS( info ) )
    !!!!
    
    !!!M.Iovine - Cholesky check:
    info = cudaDeviceSynchronize()
    IF (info /= 0) CALL lax_error__(' cdiaghg_gpu ', 'sync after Cholesky', ABS(info))
    
    IF (.NOT. ALLOCATED(dinfo_host)) ALLOCATE(dinfo_host(n_k))
    dinfo_host = d_info(1:n_k)
    print *, '[CHOLESKY d_info] per-batch status =', dinfo_host
    !!!!!

!!! DEBUGGING LINES:
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[2] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))
!!!!

    !! We assign to each element of arr_of_ptr the device pointer to each 2D slice in h_d :
    do k = 1, n_k
        arr_of_ptr_h(k) = c_devloc(h_d(:,:,k))
    end do
    
    !!! M.Iovine - we copy the c_devptr in the host array arr_of_ptr_h to the device array arr_of_ptr_h_d
    arr_of_ptr_h_d = arr_of_ptr_h


    !!!!M.Iovine - triagular pre and post multiplication of Hamiltonian:
    !! Ly = H :
    info = cublasZtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, &
           CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, &
           n, n, alpha, arr_of_ptr_s_d, ldh, arr_of_ptr_h_d, ldh, n_k)
    IF ( info /= CUBLAS_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cublasZtrsmBatched-LEFT',  ABS( info ) )

    info = cudaDeviceSynchronize() !!M.Iovine - added a synchronize
    IF (info /= 0) CALL lax_error__(' cdiaghg_gpu ', 'sync after triang left', ABS(info))

!!! DEBUGGING LINES
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[3] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))
!!!!

    !! y = w L(conj. transpose) :
    info = cublasZtrsmBatched(cublas_handle, CUBLAS_SIDE_RIGHT, &
           CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, &
           n, n, alpha, arr_of_ptr_s_d, ldh, arr_of_ptr_h_d, ldh, n_k)    
    IF ( info /= CUBLAS_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cublasZtrsmBatched-RIGT',  ABS( info ) )
      !
    
    info = cudaDeviceSynchronize() !!M.Iovine - added a synchronize
    IF (info /= 0) CALL lax_error__(' cdiaghg_gpu ', 'sync after triang right', ABS(info))

!!! DEBUGGING LINES:
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[4] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))


      !!!! M.Iovine - we define the parameters for the Jacobi algorithm corresponding to the diagonalization done through the batched cuSolver routine:
      info = cusolverDnCreateSyevjInfo(syevj_params)
      IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnCreateSyevjInfo',  ABS( info ) )
      info = cusolverDnXsyevjSetTolerance(syevj_params, 0.D0) !! The tolerance is set to the default value (0)
      IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnXsyevjSetTolerance',  ABS( info ) )
      info = cusolverDnXsyevjSetMaxSweeps(syevj_params, 100) !!! The maximum swe    eps is the default value (100)
      IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnXsyevjSetMaxSweeps',  ABS( info ) )
      info = cusolverDnXsyevjSetSortEig(syevj_params, 1) !!M.Iovine - ordering in ascending order
      IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnXsyevjSetSortEig',  ABS( info ) )
      !!!!

      !!!! M.Iovine - We change the routine from the single kernel call to the batched routine of NVIDIA Cusolver:
      info = cusolverDnZheevjBatched_bufferSize(cuSolverHandle_batched, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, &
                                               n, h_d, ldh, e_d, lwork_d, syevj_params, n_k)
      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZheevjBatched failed ', ABS( info ) )
      !
#if ! defined(__USE_GLOBAL_BUFFER)
      ALLOCATE(work_d(1*lwork_d), STAT = info)
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
      !!!! M.Iovine - We allocate the array d_info and check if it's successful :
      !ALLOCATE(d_info(n_k), STAT=info)
      !IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate d_info ', ABS( info ) )
#else
      CALL dev%lock_buffer( work_d,  lwork_d, info )
      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
#endif
      !
      !! Debugging lines:
      IF (.NOT. ALLOCATED(h_sym_chk)) ALLOCATE(h_sym_chk(n,n))
        h_sym_chk = h_d(1:n,1:n,1)
        max_asym = 0.0_DP
        DO jj_sym = 1, n
        DO ii_sym = 1, jj_sym-1
                max_asym = MAX(max_asym, ABS(h_sym_chk(ii_sym,jj_sym) - CONJG(h_sym_chk(jj_sym,ii_sym))))
          END DO
        END DO
        print *, '[SYM CHECK] max Hermitian asymmetry in h_d before eigensolver =', max_asym


      !!!! M.Iovine - We change the routine from the single kernel call to the batched routine of NVIDIA Cusolver:
      info = cusolverDnZheevjBatched(cuSolverHandle_batched, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, &
      n, h_d, ldh, e_d, work_d, lwork_d, d_info(1), syevj_params, n_k)
      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZheevjBatched failed ', ABS( info ) )
     
     info = cudaDeviceSynchronize() !!M.Iovine - added a synchronize
    IF (info /= 0) CALL lax_error__(' cdiaghg_gpu ', 'sync after batched diag.', ABS(info))
    dinfo_host = d_info(1:n_k)
    print *, '[DIAG CUSOLVER CHECK NEW d_info] per-batch status =', dinfo_host

     !!! DEBUGGING LINES: 
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[5] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))


      !! M.Iovine - we destroy the SyevjInfo object:
      info = cusolverDnDestroySyevjInfo(syevj_params)
    !!! LAST PART ADDED FOR DEBUGGING:
    do k = 1, n_k
        arr_of_ptr_h(k) = c_devloc(h_d(:,:,k))
    end do

    do k = 1, n_k
        arr_of_ptr_s(k) = c_devloc(s_d(:,:,k))
    end do
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    !!!! M.Iovine : we need to take into account the fact that the eigenvalues got after the factorization and the triangular matrix mult. are the same of the initial problem, but this is not true for the eigenvectors, so we need to solve a triangular system to get the effective eigenvectors:
    !! M.Iovine : it is important to observe that the current eigenvectors are stored in the h_d matrix: h_d = L(conj. transpose) * eigvect
    info = cublasZtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, &
           CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, &
           n, n, alpha, arr_of_ptr_s_d, ldh, arr_of_ptr_h_d, ldh, n_k)
    IF ( info /= CUBLAS_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cublasZtrsmBatched-eigenvectors',  ABS( info ) )
   
   info = cudaDeviceSynchronize() !!M.Iovine - added a synchronize
    IF (info /= 0) CALL lax_error__(' cdiaghg_gpu ', 'sync after triang final', ABS(info))

   !!! DEBUGGING LINES:   
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = h_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[6] h_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))

IF (.NOT. ALLOCATED(nan_echk)) ALLOCATE(nan_echk(n), nan_emask(n))
info = cudaDeviceSynchronize()
IF (info /= 0) CALL lax_error__('cdiaghg_gpu', &
                                 'sync before reading e_d', ABS(info))
nan_echk = e_d(1:n,1)
nan_emask = ieee_is_nan(nan_echk)
print *, '[E] e_d has_nan =', ANY(nan_emask), ' first vals =', nan_echk(1:5)

    !!!! M.Iovine - We need to order in acending way the eigenvalues
    !!!! and the corresponding eigenvectors:
    !! we add !$cuf kernel do(1) because we need to do the reordering
    !! on the device --> 
    !!$cuf kernel do(1) <<<*,*,0,laxlib_cuda_stream>>>
    !do ik = 1, n_k
     !   do b = 1, n-1
      !      ind_min = b
       !     do k = b,n
        !        if (e_d(k, ik) .le. e_d(ind_min, ik)) then
         !           ind_min = k
          !      end if
           ! end do
           ! minim = e_d(ind_min,ik)
           ! e_d(ind_min,ik) = e_d(b,ik)
           ! e_d(b,ik) = minim
            !!! We swap also the columns with indices equal to the
            !!! eigenvalues ones in order to reorder also the 
            !!! eigenvectors array!
            !do i = 1, n
             !   min_temp = h_d(i,ind_min,ik)
              !  h_d(i,ind_min,ik) = h_d(i,b,ik)
               ! h_d(i,b,ik) = min_temp
            !end do
        !end do
    !end do



!!!! M.Iovine - Modified the loops nested from 2 to 3 --> so we changed kernel do(2) to kernel do(3) :
!!$cuf kernel do(3) <<<*,*,0,laxlib_cuda_stream>>>
      !DO k=1,n_k
       !  DO j=1,n
        !    DO i=1,n
         !       IF(j <= m) v_d(i,j,k) = h_d(i,j,k) !!!!M.Iovine - the array becomes 3D and also m is an array because 
          !      h_d(i,j,k) = h_bkp_d(i,j,k)
           !     s_d(i,j,k) = s_bkp_d(i,j,k)
            !ENDDO
        ! ENDDO
      !ENDDO
!$cuf kernel do(3) <<<*,*,0,laxlib_cuda_stream>>>
DO k = 1, n_k
   DO j = 1, m
      DO i = 1, n
         v_d(i, j, k) = h_d(i, j, k)
      END DO
   END DO
END DO

!$cuf kernel do(3) <<<*,*,0,laxlib_cuda_stream>>>
DO k = 1, n_k
   DO j = 1, n
      DO i = 1, n
         h_d(i, j, k) = h_bkp_d(i, j, k)
         s_d(i, j, k) = s_bkp_d(i, j, k)
      END DO
   END DO
END DO


info = cudaDeviceSynchronize()
 IF (info /= 0) CALL lax_error__('cdiaghg_gpu', &
                                 'sync before reading e_d', ABS(info))

!
      !
      ! Do not destroy the handle to save the (re)creation time on each call.
      !
      !info = cusolverDnDestroy(cuSolverHandle_batched)
      !IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu_batched ', ' cusolverDnDestroy_batched failed ', ABS( info ) )
      !

 !!! DEBUGGING LINES:   
IF (.NOT. ALLOCATED(nan_chk)) ALLOCATE(nan_chk(n,n), nan_mask_r(n,n), nan_mask_i(n,n), &
                                        inf_mask_r(n,n), inf_mask_i(n,n))
nan_chk = v_d(1:n,1:n,1)
nan_mask_r = ieee_is_nan(REAL(nan_chk))
nan_mask_i = ieee_is_nan(AIMAG(nan_chk))
inf_mask_r = (.NOT. ieee_is_finite(REAL(nan_chk))) .AND. (.NOT. nan_mask_r)
inf_mask_i = (.NOT. ieee_is_finite(AIMAG(nan_chk))) .AND. (.NOT. nan_mask_i)
has_nan_dbg = ANY(nan_mask_r) .OR. ANY(nan_mask_i)
has_inf_dbg = ANY(inf_mask_r) .OR. ANY(inf_mask_i)
print *, '[7] v_d input has_nan =', has_nan_dbg, ' has_inf =', has_inf_dbg, &
         ' maxabs =', MAXVAL(ABS(nan_chk), MASK = ieee_is_finite(REAL(nan_chk)) .AND. ieee_is_finite(AIMAG(nan_chk)))
      
      
      
#if ! defined(__USE_GLOBAL_BUFFER)
      DEALLOCATE(work_d)
      DEALLOCATE(h_bkp_d, s_bkp_d)
      DEALLOCATE(d_info) !!!! M.Iovine - we deallocate d_info
      DEALLOCATE(arr_of_ptr_h) !! M.Iovine - we deallocate arr_of_ptr
      DEALLOCATE(arr_of_ptr_s) !! M.Iovine - we deallocate arr_of_ptr
      DEALLOCATE(arr_of_ptr_h_d) !! M.Iovine - we deallocate arr_of_ptr
      DEALLOCATE(arr_of_ptr_s_d) !! M.Iovine - we deallocate arr_of_ptr
#else
      CALL dev%release_buffer( work_d,  info )
      CALL dev%release_buffer( h_bkp_d, info )
      CALL dev%release_buffer( s_bkp_d, info )
#endif
!      IF (ALLOCATED(h_sym_chk)) DEALLOCATE(h_sym_chk) !!M.IOvine - debugging lines
!! We destroy the cublas handle:
!#if defined(__CUDA)
 !   istat_cublas = cublasDestroy(cublas_handle)
!#endif

      !
      ! Keeping compatibility for both CUSolver and CustomEigensolver, CustomEigensolver below
      !
#else
     CALL lax_error__( 'cdiaghg', 'Called GPU eigensolver without GPU support', 1 )
#endif

!! We destroy the cublas handle:
#if defined(__CUDA)
    istat_cublas = cublasDestroy(cublas_handle)
#endif

info = cudaDeviceSynchronize()
 IF (info /= 0) CALL lax_error__('cdiaghg_gpu', &
                                 'sync before reading e_d', ABS(info))

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
  !!!! M.Iovine - We add a loop to take into account the k-points of the batch - we need to make it be executed on the device :
  !$cuf kernel do(3) <<<*,*,0,laxlib_cuda_stream>>>
  DO k=1,n_k 
    CALL MPI_BCAST( e_d(:,k), n, MPI_DOUBLE_PRECISION, root_bgrp, intra_bgrp_comm, info )
    IF ( info /= 0 ) &
            CALL lax_error__( 'cdiaghg', 'error broadcasting array e_d', ABS( info ) )
    CALL MPI_BCAST( v_d(:,:,k), ldh*m, MPI_DOUBLE_COMPLEX, root_bgrp, intra_bgrp_comm, info )
    IF ( info /= 0 ) &
            CALL lax_error__( 'cdiaghg', 'error broadcasting array v_d', ABS( info ) )
    info = cudaDeviceSynchronize() ! this is probably redundant...
    IF ( info /= 0 ) &
            CALL lax_error__( 'cdiaghg', 'error synchronizing device (second)', ABS( info ) )
  END DO  
#else
  !!!! M.Iovine - We add a loop to take into account the k-points of th batch :
  ALLOCATE(e_h(n, n_k), v_h(ldh, m, n_k))
  DO k=1,n_k
    e_h(1:n, k) = e_d(1:n, k)
    v_h(1:ldh, 1:m, k) = v_d(1:ldh, 1:m, k)
    CALL MPI_BCAST( e_h(:,k), n, MPI_DOUBLE_PRECISION, root_bgrp, intra_bgrp_comm, info )
    IF ( info /= 0 ) &
            CALL lax_error__( 'cdiaghg', 'error broadcasting array e_d', ABS( info ) )
    CALL MPI_BCAST( v_h(:,:,k), ldh*m, MPI_DOUBLE_COMPLEX, root_bgrp, intra_bgrp_comm, info )
    IF ( info /= 0 ) &
            CALL lax_error__( 'cdiaghg', 'error broadcasting array v_d', ABS( info ) )
    e_d(1:n, k) = e_h(1:n, k)
    v_d(1:ldh, 1:m, k) = v_h(1:ldh, 1:m, k)
  END DO
  DEALLOCATE(e_h, v_h)
#endif
#endif
  !
  CALL stop_clock_gpu( 'cdiaghg' )
  !
  
  !!!DEBUGGING LINES:
  IF (.NOT. ALLOCATED(nan_echk)) ALLOCATE(nan_echk(n), nan_emask(n))
nan_echk = e_d(1:n,1)
nan_emask = ieee_is_nan(nan_echk)
print *, '[E2] e_d has_nan =', ANY(nan_emask), ' first vals =', nan_echk(1:5)

!!!! DEBUGGING LINES:  
#if defined(__CUDA)
        IF (ALLOCATED(nan_chk)) DEALLOCATE(nan_chk, nan_mask_r, nan_mask_i, inf_mask_r, inf_mask_i)
        IF (ALLOCATED(nan_echk)) DEALLOCATE(nan_echk, nan_emask)
#endif
  RETURN
  !
END SUBROUTINE laxlib_cdiaghg_gpu_batched
