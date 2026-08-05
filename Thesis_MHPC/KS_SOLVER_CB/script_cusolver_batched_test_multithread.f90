PROGRAM script_diag

USE cudafor !!! Cuda Fortran
USE cusolverDn
USE openacc
USE iso_c_binding, ONLY: c_ptr, c_null_ptr !! We need the binding to pass C-pointers and the C-pointer to null --> IN THIS WAY WE CAN AVOID TOUSE DERIVED TYPE for the info parameter of cudasolverDn

IMPLICIT NONE


INTEGER:: i, n=3, num_k=4 ! We have num_k matrices, one for each k-point

!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS

! In this case I use an allocatable, not a pointer!!
!COMPLEX(8) ALLOCATABLE:: general_matr(:,:,:) ! Array 3D for multistream case

!COMPLEX(8) ALLOCATABLE:: matr(:,:) ! Array 2D for the single threads

COMPLEX(8), ALLOCATABLE:: matr3D(:,:,:) ! Array 3D for storing all the matrices in one matrix for testing the CuSolvr Batched for 1 or multiple threads/streams!!

REAL(8), ALLOCATABLE :: d_w(:,:) ! Stores the eigenvalues of every array matrix




!!!! WE MOVE THE FOLLOWING LINE AT THE BEGINNING, SO IN THIS WE CAN DEFINE THE ARRAYS SIZE AND THE NUMBER OF ARRAYS AT THE COMPILE TIME AND SAVE 
!!!! EXECUTION TIME!!!!
!INTEGER: i, n=3, num_k ! We have num_k matrices, one for each k-point



!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

COMPLEX(8), ALLOCATABLE :: d_work(:)
INTEGER :: lwork !!! work index corresponding to the work???
!TYPE(syevjInfo_t) :: syevj_params !!!! Parameters for the standard case solution through Jacobi algorithm 
TYPE(c_ptr) :: syevj_params = c_null_ptr
INTEGER, ALLOCATABLE :: d_info(:)

!#if defined(_OPENMP)
 ! USE omp_lib, only: omp_get_thread_num
!#endif


!numthreads = num_k

! Allocation on the host
!ALLOCATE(general_matr(n, n, num_threads))

!! Allocation on host of the matrix for the test on the single thread!!:
ALLOCATE(matr3D(n,n,num_k))


!! We allocate d_info:
ALLOCATE(d_info(num_k))

!! We allocate d_w :
ALLOCATE(d_w(n,num_k))


!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
do i = 1, num_k
    !ALLOCATE(matr(n,n))

    matr3D(1,1,i) = (2.0D0, 0.0D0)
    matr3D(2,2,i) = (2.0D0, 0.0D0)
    matr3D(3,3,i) = (2.0D0, 0.0D0)
    matr3D(1,2,i) = (0.0D0, 0.0D0)
    matr3D(1,3,i) = (0.0D0, -1.0D0)
    matr3D(2,3,i) = (0.0D0, 0.0D0)
    matr3D(3,1,i) = (0.0D0, 1.0D0)
    matr3D(3,2,i) = (0.0D0, 0.0D0)
    matr3D(2,1,i) = (0.0D0, 0.0D0)
    
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
!!$omp end parallel


!! We copy the general matrix on the GPU:
!!$acc enter data copyin(general_matr)


!! We copy the matrix 3D for testing on single thread from host to gpu:
!$acc enter data copyin(matr3D)

!!! WARNING:
!!! It is not necessary to copy scalar variables from host to device, right??

!! We copy d_w from host to device:
!$acc enter data copyin(d_w)


!!  We copy d_info from host to device:
!$acc enter data copyin(d_info)

!! Parameters for the Jacobi algorithm:
call cusolverDnCreateSyevjInfo(syevj_params)
call cusolverDnXsyevjSetTolerance(syevj_params, 0.D0) !! The tolerance is set to the default value (0)
call cusolverDnXsyevjSetMaxSweeps(syevj_params, 100) !!! The maximum sweeps is the default value (100)
call cusolverDnXsyevjSetSortEig(syevj_params, 0) !!!! Disable the sorting of the eigenvalues!!


!!! WE use the Batched version of CuSolver: cusolverDnZheevjBatched
!!! It works only for the standard (no overlap case) 

STATUS = cusolverDnCreate(cusolver_handle)
!IF (info /= CUSOLVER_STATUS_SUCCESS) PRINT *, 'Error cuSolverDnCreate'


!!! Before running the Batched solver API, we need to retrieve/define
!!! a proper buffer for it:
!!! Helper functions of the type bufferSize calculate the sizes needed for pre-allocated buffer
call cusolverDnZheevjBatched_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, matr3D, n, d_w, lwork, syevj_params, num_k)

!!! We allocate d_work after having run Buffersize:
ALLOCATE(d_work(lwork))

!! We copy d_work from host to device:
!$acc enter data copyin(d_work)


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr3D, d_w, d_work, d_info)
call cusolverDnZheevjBatched(cusolver_handle, &
    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, &
    n, matr3D, n, d_w, d_work, lwork, d_info, syevj_params, num_k)
!$acc end host_data


DEALLOCATE(matr3D)
DEALLOCATE(d_w)

!!!! We deallocate the handles created:
call cusolverDnDestroySyevjInfo(syevj_params)
STATUS =  cusolverDnDestroy(cusolver_handle)


END PROGRAM script_diag
