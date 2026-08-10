PROGRAM: script_diag

IMPLICIT NONE
USE openacc 
USE iso_c_binding, ONLY: c_ptr !! We need the binding to pass arrays of pointers --> the acc_deviceptr gives a memory address wtitten in c format

! In this case I use an allocatable, not a pointer!!
COMPLEX(8) ALLOCATABLE: general_matr(:,:,:) ! Array 3D

COMPLEX(8) ALLOCATABLE: matr(:,:) ! Array 2D for the single threads

REAL(8), ALLOCATABLE : d_w(:,:) ! Stores the eigenvalues of every array matrix 


INTEGER: i, n=3, num_k ! We have num_k matrices, one for each k-point

#if defined(_OPENMP)
  USE omp_lib, only: omp_get_thread_num
#endif


numthreads = num_k

! Allocation on the host
ALLOCATE(general_matr(n, n, num_threads))


!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!$omp do
do i = 1, num_k
    ALLOCATE(matr(n,n))
    CALL populate_array(n, general_matr)
    !matr(1,1) = COMPLEX(2.0D0, 0.0D0, kind=8)
    !matr(2,2) = COMPLEX(2.0D0, 0.0D0, kind=8)
    !matr(3,3) = COMPLEX(2.0D0, 0.0D0, kind=8)
    !matr(1,2) = COMPLEX(0.0D0, 0.0D0, kind=8)
    !matr(1,3) = COMPLEX(0.0D0, -1.0D0, kind=8)
    !matr(2,3) = COMPLEX(0.0D0, 0.0D0, kind=8)
    !matr(3,1) = COMPLEX(0.0D0, 1.0D0, kind=8)
    !matr(3,2) = COMPLEX(0.0D0, 0.0D0, kind=8)
    !matr(2,1) = COMPLEX(0.0D0, 0.0D0, kind=8)
    
    !! WE save the current matrix on the thread in the general array
    !general_matr(:,:,i) = matr

end do
!$omp end parallel


!! We copy the general matrix on the GPU:
!$acc enter data copyin(general_matr)


!!! WE use the Batched version of CuSolver: cusolverDnZheevjBatched
!!! It works only for the standard (no overlap case) 

!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!

!$acc host_data use_device(general_matr, d_w, d_work, d_info)
call cusolverDnZheevjBatched(cuSolverHandle, &
    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, &
    n, general_matr, n, d_w, d_work, lwork, d_info, params, num_threads)
!$acc end host_data

