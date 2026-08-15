PROGRAM script_diag

USE cudafor !!! Cuda Fortran
USE cusolverDn
USE openacc
USE iso_c_binding, ONLY: c_ptr, c_null_ptr !! We need the binding to pass C-pointers and the C-pointer to null --> IN THIS WAY WE CAN AVOID TOUSE DERIVED TYPE for the info parameter of cudasolverDn

IMPLICIT NONE


INTEGER:: i, n=6, num_k=4 ! We have num_k matrices, one for each k-point

!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS

!! Output variable of the Batched kernel calls:
INTEGER, ALLOCATABLE :: STATUS_B(:)

! In this case I use an allocatable, not a pointer!!
!COMPLEX(8) ALLOCATABLE:: general_matr(:,:,:) ! Array 3D for multistream case

!COMPLEX(8) ALLOCATABLE:: matr(:,:) ! Array 2D for the single threads

COMPLEX(8), ALLOCATABLE:: matr2D(:,:) ! Array 3D for storing all the matrices in one matrix for testing the CuSolvr Batched for 1 or multiple threads/streams!!

COMPLEX(8), ALLOCATABLE:: s_w(:,:) ! Overlap matrix array!!

REAL(8), ALLOCATABLE :: d_w(:) ! Stores the eigenvalues of every array matrix



!!!! WE MOVE THE FOLLOWING LINE AT THE BEGINNING, SO IN THIS WE CAN DEFINE THE ARRAYS SIZE AND THE NUMBER OF ARRAYS AT THE COMPILE TIME AND SAVE 
!!!! EXECUTION TIME!!!!
!INTEGER: i, n=3, num_k ! We have num_k matrices, one for each k-point



!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

COMPLEX(8), ALLOCATABLE :: d_work(:)
INTEGER :: lwork !!! work index corresponding to the work???
!TYPE(cusolverDnSyevjInfo) :: syevj_params !!!! Parameters for the standard case solution through Jacobi algorithm 
!! NVIDIA already wrapped the opaque c pointer in the Fortran module
!TYPE(c_ptr) :: syevj_params = c_null_ptr
! Despite in C the d_info is considered a pointer, we checked by testing the code that probably in the interface contained in the
! cusolverDn module the argument is treated as a simple scalar
!! h_meig is an output of the cuSolver NVIDIA routine!!
INTEGER :: d_info, h_meig

!! Variables for printing results:
INTEGER :: c, j, a, b


!#if defined(_OPENMP)
 ! USE omp_lib, only: omp_get_thread_num
!#endif


!numthreads = num_k

! Allocation on the host
!ALLOCATE(general_matr(n, n, num_threads))

!! Allocation on host of the matrix for the test on the single thread!!:
ALLOCATE(matr2D(n,n))

!! Overlap array matrix allocation:
ALLOCATE(s_w(n,n))

!! We allocate d_w :
ALLOCATE(d_w(n))

!! We allcate the array output for Batched kernel calls:
ALLOCATE(STATUS_B(num_k))


!!! Matr3D initialization:
matr2D = (0.D0, 0.D0)

!! The overlap matrix in this case is the identity matrix:
s_w = (0.D0,0.D0)
do i = 1, n 
    do j = 1, n
        if (j .eq. i) s_w(i,j) = (1.D0,0.D0)
    end do
end do

!!! Eigenvalue matrix initialization:
d_w = (0.D0)

!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
!do i = 1, num_k
    !ALLOCATE(matr(n,n))
    
    !! Case with n = 3 :
    !matr2D(1,1) = (2.0D0, 0.0D0)
    !matr2D(2,2) = (2.0D0, 0.0D0)
    !matr2D(3,3) = (2.0D0, 0.0D0)
    !matr2D(1,2) = (0.0D0, 0.0D0)
    !matr2D(1,3) = (0.0D0, -1.0D0)
    !matr2D(2,3) = (0.0D0, 0.0D0)
    !matr2D(3,1) = (0.0D0, 1.0D0)
    !matr2D(3,2) = (0.0D0, 0.0D0)
    !matr2D(2,1) = (0.0D0, 0.0D0)
    
    !matr2D(1,1) = (1.0d0,0.0d0)
    !matr2D(2,2) = (2.0d0,0.0d0)
    !matr2D(3,3) = (3.0d0,0.0d0)
    !matr2D(4,4) = (4.0d0,0.0d0)
    !matr2D(5,5) = (5.0d0,0.0d0)
    !matr2D(6,6) = (6.0d0,0.0d0)
    
    !! Case with n = 6 :
    matr2D(1,1) = (1.0D0, 1.0D0)
    matr2D(2,2) = (2.0D0, -1.0D0)
    matr2D(3,3) = (3.0D0, 2.0D0)
    matr2D(4,4) = (-1.0D0, 1.0D0)
    matr2D(5,5) = (4.0D0, -2.0D0)
    matr2D(6,6) = (5.0D0, 1.0D0)
    matr2D(1,2) = (2.0D0, -1.0D0)
    matr2D(1,3) = (0.0D0, 0.0D0)
    matr2D(1,4) = (0.0D0, 0.0D0)
    matr2D(1,5) = (1.0D0, 0.0D0)
    matr2D(1,6) = (0.0D0, -1.0D0)
    matr2D(2,1) = (0.0D0, 0.0D0)
    matr2D(2,3) = (1.0D0, 1.0D0)
    matr2D(2,4) = (0.0D0, 0.0D0)
    matr2D(2,5) = (0.0D0, 0.0D0)
    matr2D(2,6) = (2.0D0, 0.0D0)
    matr2D(3,1) = (1.0D0, 0.0D0)
    matr2D(3,2) = (0.0D0, 0.0D0)
    matr2D(3,4) = (0.0D0, -1.0D0)
    matr2D(3,5) = (0.0D0, 0.0D0)
    matr2D(3,6) = (0.0D0, 0.0D0)
    matr2D(4,1) = (0.0D0, 0.0D0)
    matr2D(4,2) = (1.0D0, 0.0D0)
    matr2D(4,3) = (0.0D0, 1.0D0)
    matr2D(4,5) = (2.0D0, 0.0D0)
    matr2D(4,6) = (0.0D0, 0.0D0)
    matr2D(5,1) = (2.0D0, 0.0D0)
    matr2D(5,2) = (0.0D0, 0.0D0)
    matr2D(5,3) = (0.0D0, 0.0D0)
    matr2D(5,4) = (0.0D0, -1.0D0)
    matr2D(5,6) = (1.0D0, 0.0D0)
    matr2D(6,1) = (0.0D0, 0.0D0)
    matr2D(6,2) = (1.0D0, 0.0D0)
    matr2D(6,3) = (2.0D0, 0.0D0)
    matr2D(6,4) = (0.0D0, 0.0D0)
    matr2D(6,5) = (1.0D0, 1.0D0)
    !! WE save the current matrix on the thread in the general array
    !general_matr(:,:,i) = matr

!end do
!!$omp end parallel


!! We copy the general matrix on the GPU:
!!$acc enter data copyin(general_matr)


!! We copy the matrix 2D for testing on single thread from host to gpu:
!$acc enter data copyin(matr2D)

!! We copy matrix overlap from host to device:
!$acc enter data copyin(s_w)

!!! WARNING:
!!! It is not necessary to copy scalar variables from host to device, right??

!! We copy d_w from host to device:
!$acc enter data copyin(d_w)


!! We copy d_info from host to device:
!$acc enter data copyin(d_info)


STATUS = cusolverDnCreate(cusolver_handle)
print *, "Status: ", STATUS

!! Parameters for the Jacobi algorithm:
!STATUS = cusolverDnCreateSyevjInfo(syevj_params)
!print *, "Status: ", STATUS
!STATUS = cusolverDnXsyevjSetTolerance(syevj_params, 0.D0) !! The tolerance is set to the default value (0)
!print *, "Status: ", STATUS
!STATUS = cusolverDnXsyevjSetMaxSweeps(syevj_params, 100) !!! The maximum sweeps is the default value (100)
!print *, "Status: ", STATUS
!STATUS = cusolverDnXsyevjSetSortEig(syevj_params, 0) !!!! Disable the sorting of the eigenvalues!!
!print *, "Status: ", STATUS

!!! WE use the Batched version of CuSolver: cusolverDnZheevjBatched
!!! It works only for the standard (no overlap case) 


!!! Before running the Batched solver API, we need to retrieve/define
!!! a proper buffer for it:
!!! Helper functions of the type bufferSize calculate the sizes needed for pre-allocated buffer
!!! s_w is the overlap matrix array!
!$acc host_data use_device(matr2D, d_w, s_w)
STATUS = cusolverDnZhegvdx_bufferSize(cusolver_handle, &
         CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,& 
         CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, &
         n, matr2D, n, s_w, n, 0.D0, 0.D0, 1, n, h_meig, d_w, lwork)
!$acc end host_data

print *, "bufferSize STATUS =", STATUS, " lwork =", lwork


!!! We allocate d_work after retrieved the Buffersize with the previousfunction:
ALLOCATE(d_work(lwork))

!! We copy d_work from host to device:
!!$acc enter data copyin(d_work)
!$acc enter data create(d_work)


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr2D, d_w, s_w, d_work)
STATUS = cusolverDnZhegvdx(cusolver_handle, & 
        CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, &
        CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER, n, &
        matr2D, n, s_w, n, 0.D0, 0.D0, 1, n, h_meig, d_w, d_work, lwork, d_info)
!$acc end host_data

!! We update the arrays on the host with the new values got on the device:
!!$acc update host(matr3D, d_w, d_info)
!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr2D, d_w, s_w, d_work)


!!! Print check:
print *, "STATUS=success"

print *, d_info

!!! Matrices print:
print *, "Matrix 2D"
do j = 1, n
    print *, matr2D(j,:)
end do

print *, "Eigenvalues"
print *, d_w(:)

!! We print the eigenvalues in a file:
OPEN(unit=20, FILE="matrix_no_batched.dat", STATUS="REPLACE")
    do j = 1,n
        WRITE(20,*) d_w(j)
    end do
CLOSE(20)


DEALLOCATE(matr2D)
DEALLOCATE(d_w)
DEALLOCATE(s_w)
DEALLOCATE(STATUS_B)
!DEALLOCATE(d_info)
DEALLOCATE(d_work) !!! We could allocate d_work only on the device throughb proper CUDA commands, but we allocated it also on the host

!!!! We deallocate the handles created:
STATUS = cusolverDnDestroy(cusolver_handle)


END PROGRAM script_diag
