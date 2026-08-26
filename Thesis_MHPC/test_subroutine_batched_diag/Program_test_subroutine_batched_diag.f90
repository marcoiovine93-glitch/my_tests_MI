!! Program to test the functionality of the subroutine introduced for the batched diagonalization of the reduced Hamiltonian:
PROGRAM script_diag_test
#if defined(__CUDA)    
    use cublas
    use cudafor
#endif
   USE util_param,    ONLY : DP
   USE mytime,          ONLY : clock_thread, clock_cuda_stream, cegterg_locker !Fixed: import thread private varibale
   USE openacc,         ONLY : acc_get_cuda_stream
 #if defined(__CUDA)
    use openacc,   only: acc_get_cuda_stream
    use laxlib_cusolver_handles, ONLY : initialize_cusolver_handles, initialize_laxlib_cuda_stream
 #endif
   !
   IMPLICIT NONE
   !
   include 'laxlib.fh'

   INTEGER :: i_batch=1, n=10, m=10, ldh=10
   INTEGER :: a, i, j, k
   COMPLEX(DP), ALLOCATABLE :: h_d(:,:,:), s_d(:,:,:), v_d(:,:,:) !! M.Iovine - added a dimension -It is still a static array
   !! matrix to be diagonalized, allocated on the GPU
   !$acc declare device_resident(h_d, s_d, v_d)
   COMPLEX(DP), ALLOCATABLE :: h(:,:), s(:,:), v(:,:)
   REAL(DP), ALLOCATABLE :: e_d(:,:)
   !$acc declare device_resident(e_d)
   REAL(DP), ALLOCATABLE :: e(:)
   INTEGER  :: info
   COMPLEX(DP) :: proj, ip
   REAL(DP)    :: nrm
   REAL(DP), ALLOCATABLE :: lambda(:)
   COMPLEX(DP), ALLOCATABLE :: Sv(:)
   COMPLEX(DP), ALLOCATABLE :: Siv(:)
   COMPLEX(DP), ALLOCATABLE :: Sjv(:)
#if defined(__CUDA)
    do i = 1, i_batch
       clock_thread = i_
       clock_cuda_stream = acc_get_cuda_stream(clock_thread)
       call initialize_laxlib_cuda_stream(clock_cuda_stream, clock_thread)
       print '("Initialized default stream in thread ",2I5,I24)', omp_get_thread_num(), clock_thread, clock_cuda_stream
    end do
 #endif

    !! Allocation of the arrays:
    ALLOCATE(h_d(ldh,n,n_k))
    ALLOCATE(s_d(ldh,n,n_k))
    ALLOCATE(v_d(ldh,n,n_k))
    ALLOCATE(e(n))
    ALLOCATE(e_d(n,n_k))
    ALLOCATE(s(n,n))
    ALLOCATE(h(n,n))
    ALLOCATE(v(n,n))
    ALLOCATE(Sv(n))
    ALLOCATE(Siv(n))
    ALLOCATE(Sjv(n))
    ALLOCATE(lambda(n))

    !!We define S as symmetric (Hermitian) positive definite and diagonally dominant :
    s = (0.D0,0.D0)

    do i = 1, n 
        s(i,i) = (5.0D0, 0.D0)
    end do
    
    do i = 1, (n-1)
        s(i,i+1) = CMPLX(0.15D0, 0.05D0, KIND=DP)
        s(i+1,i) = CONJG(s(i,i+1))
    end do
    
    do i = 1, (n-2)
        s(i,i+2) = CMPLX(0.05D0, -0.02D0, KIND=DP)
        s(i+2,i) = CONJG(s(i,i+2))
    end do
    !!!!

    v = (0.D0,0.D0) !!! We initalize the starting vectors
    do k = 1, np
        v(k,k)   = (1.D0,0.D0)  !seed with distinct unit directions
        v(k+1,k) = (0.5D0,0.D0) !plus overlap to make GS nontrivial
    end do
    
    DO k = 1, n
        !! Orthogonalization
        DO j = 1, k-1
            proj = SUM(CONJG(v(:,j)) * MATMUL(s, v(:,k)))
            v(:,k) = v(:,k) - proj * v(:,j)
        END DO
        !! Normalization :
        Sv = MATMUL(s, v(:,k))
        nrm = SQRT(REAL(SUM(CONJG(v(:,k))*Sv), DP))
        v(:,k) = v(:,k) / nrm
    END DO

    ! sanity check: V^H S V should be identity to ~1e-14
    PRINT *, '--- V^H S V (should be identity) ---'
    DO i = 1, np
       DO j = 1, np
          ip = SUM(CONJG(v(:,i)) * MATMUL(s, v(:,j)))
          PRINT '(A,I2,I2,A,2ES14.6)', ' (', i, j, ') = ', ip
       END DO
    END DO

    !eigenvalues
    lambda(1) = 1.0D0
    lambda(2) = 3.5D0
    lambda(3) = 7.0D0
    lambda(4) = 9.0D0
    lambda(5) = 11.0D0
    lambda(6) = 13.0D0
    lambda(7) = 15.0D0
    lambda(8) = 17.0D0
    lambda(9) = 19.0D0
    lambda(10) = 21.0D0

    !!!!!!!!!! We build h :
    h = (0.D0,0.D0)
    DO i = 1, n
       DO j = 1, n
          DO k = 1, np
             h(i,j) = h(i,j) + lambda(k) * &
                    SUM(s(i,:) * v(:,k)) * CONJG(SUM(s(j,:) * v(:,k)))
          END DO
       END DO
    END DO
    !!!!!!!!!!!
    
    !! We copy the h and s 2d arrays on the device:
    !$acc copyin(h, s, v)
    !$acc parallel loop collapse(2) present(h_d, s_d, v_d, h, s, v)
    do i = 1, n
       do j = 1, n
           h_d(j,i,i_batch) = h(j,i)
           s_d(j,i,i_batch) = s(j,i)
           v_d(j,i,i_batch) = v(j,i)
       end do
    end do


    info = diaghg(n,m,h_d,s_d,ldh,e_d,v_d,n_k,0,0,0)
    
    !!!! We at first copy e to the device:
    !$acc data copyin(e)
    !$acc parallel loop present(e, e_d)
    DO i = 1, n
       e(i) = e_d(i, i_batch)
    END DO

    !$acc parallel loop collapse(2) present(v, v_d)
    DO j = 1, n
        DO i = 1, n
            v(i, j) = v_d(i, j, i_batch)
        END DO
    END DO

    PRINT *, 'Computed eigenvalues:', e
    PRINT *, 'Expected among them:', lambda
    
    DO i = 1, n
       DO k = 1, n
          IF (ABS(e(i) - lambda(k)) < 1.D-8) THEN
             ip = SUM(CONJG(v(:,i)) * MATMUL(s, v(:,k)))
             PRINT '(A,I2,A,ES14.6)', ' |<v_computed|S|v_planted_', k, '>| =', ABS(ip)
          END IF
       END DO
    END DO

    PRINT *, '--- V_computed^H S V_computed diag check ---'
    DO i = 1, n
       ip = SUM(CONJG(v(:,i)) * MATMUL(s, v(:,i)))
       PRINT '(A,I3,A,2ES14.6)', ' vec ', i, ' S-norm = ', ip
    END DO
  

    !! Deallocation of the arrays :
     DEALLOCATE(h_d)
     DEALLOCATE(s_d)
     DEALLOCATE(v_d)
     DEALLOCATE(e)
     DEALLOCATE(e_d)
     DEALLOCATE(s)
     DEALLOCATE(h)
     DEALLOCATE(v)
     DEALLOCATE(Sv)
     DEALLOCATE(Siv)
     DEALLOCATE(Sjv)
     DEALLOCATE(lambda)

END PROGRAM script_diag_test
