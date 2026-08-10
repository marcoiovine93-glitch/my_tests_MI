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

COMPLEX(8), ALLOCATABLE:: matr3D(:,:,:) ! Array 3D for storing all the matrices in one matrix for testing the CuSolvr Batched for 1 or multiple threads/streams!!

REAL(8), ALLOCATABLE :: d_w(:,:) ! Stores the eigenvalues of every array matrix


!!! We define an array to print the eigenvalues ordered:
REAL(8), ALLOCATABLE :: d_w_ord(:)
REAL(8) :: minim
INTEGER(8) :: indexm



!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

COMPLEX(8), ALLOCATABLE :: d_work(:)
INTEGER :: lwork !!! work index corresponding to the work???
TYPE(cusolverDnSyevjInfo) :: syevj_params !!!! Parameters for the standard case solution through Jacobi algorithm 
!! NVIDIA already wrapped the opaque c pointer in the Fortran module
!TYPE(c_ptr) :: syevj_params = c_null_ptr
! Despite in C the d_info is considered a pointer, we checked by testing the code that probably in the interface contained in the
! cusolverDn module the argument is treated as a simple scalar
INTEGER, ALLOCATABLE :: d_info(:)

!! Variables for printing results:
INTEGER :: c, j, a, b, k


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

!! Ordered eigenvalues allocation:
ALLOCATE(d_w_ord(n))

!! We allcate the array output for Batched kernel calls:
ALLOCATE(STATUS_B(num_k))


!!! Matr3D initialization:
matr3D = (0.D0, 0.D0)

!!! Eigenvalue matrix initialization:
d_w = (0.D0, 0.D0)

!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
do i = 1, num_k
    !ALLOCATE(matr(n,n))
    
    !! Case with n = 3 :
    !matr3D(1,1,i) = (2.0D0, 0.0D0)
    !matr3D(2,2,i) = (2.0D0, 0.0D0)
    !matr3D(3,3,i) = (2.0D0, 0.0D0)
    !matr3D(1,2,i) = (0.0D0, 0.0D0)
    !matr3D(1,3,i) = (0.0D0, -1.0D0)
    !matr3D(2,3,i) = (0.0D0, 0.0D0)
    !matr3D(3,1,i) = (0.0D0, 1.0D0)
    !matr3D(3,2,i) = (0.0D0, 0.0D0)
    !matr3D(2,1,i) = (0.0D0, 0.0D0)
    
    !! Case with n = 6 :
    matr3D(1,1,i) = (1.0D0, 1.0D0)
    matr3D(2,2,i) = (2.0D0, -1.0D0)
    matr3D(3,3,i) = (3.0D0, 2.0D0)
    matr3D(4,4,i) = (-1.0D0, 1.0D0)
    matr3D(5,5,i) = (4.0D0, -2.0D0)
    matr3D(6,6,i) = (5.0D0, 1.0D0)
    matr3D(1,2,i) = (2.0D0, -1.0D0)
    matr3D(1,3,i) = (0.0D0, 0.0D0)
    matr3D(1,4,i) = (0.0D0, 0.0D0)
    matr3D(1,5,i) = (1.0D0, 0.0D0)
    matr3D(1,6,i) = (0.0D0, -1.0D0)
    matr3D(2,1,i) = (0.0D0, 0.0D0)
    matr3D(2,3,i) = (1.0D0, 1.0D0)
    matr3D(2,4,i) = (0.0D0, 0.0D0)
    matr3D(2,5,i) = (0.0D0, 0.0D0)
    matr3D(2,6,i) = (2.0D0, 0.0D0)
    matr3D(3,1,i) = (1.0D0, 0.0D0)
    matr3D(3,2,i) = (0.0D0, 0.0D0)
    matr3D(3,4,i) = (0.0D0, -1.0D0)
    matr3D(3,5,i) = (0.0D0, 0.0D0)
    matr3D(3,6,i) = (0.0D0, 0.0D0)
    matr3D(4,1,i) = (0.0D0, 0.0D0)
    matr3D(4,2,i) = (1.0D0, 0.0D0)
    matr3D(4,3,i) = (0.0D0, 1.0D0)
    matr3D(4,5,i) = (2.0D0, 0.0D0)
    matr3D(4,6,i) = (0.0D0, 0.0D0)
    matr3D(5,1,i) = (2.0D0, 0.0D0)
    matr3D(5,2,i) = (0.0D0, 0.0D0)
    matr3D(5,3,i) = (0.0D0, 0.0D0)
    matr3D(5,4,i) = (0.0D0, -1.0D0)
    matr3D(5,6,i) = (1.0D0, 0.0D0)
    matr3D(6,1,i) = (0.0D0, 0.0D0)
    matr3D(6,2,i) = (1.0D0, 0.0D0)
    matr3D(6,3,i) = (2.0D0, 0.0D0)
    matr3D(6,4,i) = (0.0D0, 0.0D0)
    matr3D(6,5,i) = (1.0D0, 1.0D0)
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


!! We copy d_info from host to device:
!$acc enter data copyin(d_info)


STATUS = cusolverDnCreate(cusolver_handle)
print *, "Status: ", STATUS

!! Parameters for the Jacobi algorithm:
STATUS = cusolverDnCreateSyevjInfo(syevj_params)
print *, "Status: ", STATUS
STATUS = cusolverDnXsyevjSetTolerance(syevj_params, 0.D0) !! The tolerance is set to the default value (0)
print *, "Status: ", STATUS
STATUS = cusolverDnXsyevjSetMaxSweeps(syevj_params, 100) !!! The maximum sweeps is the default value (100)
print *, "Status: ", STATUS
STATUS = cusolverDnXsyevjSetSortEig(syevj_params, 0) !!!! Disable the sorting of the eigenvalues!!
print *, "Status: ", STATUS

!!! WE use the Batched version of CuSolver: cusolverDnZheevjBatched
!!! It works only for the standard (no overlap case) 


!!! Before running the Batched solver API, we need to retrieve/define
!!! a proper buffer for it:
!!! Helper functions of the type bufferSize calculate the sizes needed for pre-allocated buffer
!$acc host_data use_device(matr3D, d_W)
STATUS = cusolverDnZheevjBatched_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, matr3D, n, d_w, lwork, syevj_params, num_k)
!$acc end host_data

print *, "bufferSize STATUS =", STATUS, " lwork =", lwork


!!! We allocate d_work after retrieved the Buffersize with the previousfunction:
ALLOCATE(d_work(lwork))

!! We copy d_work from host to device:
!!$acc enter data copyin(d_work)
!$acc enter data create(d_work)


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr3D, d_w, d_work, d_info)
STATUS = cusolverDnZheevjBatched(cusolver_handle, &
    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, &
    n, matr3D, n, d_w, d_work, lwork, d_info(1), syevj_params, num_k)
!$acc end host_data

!! We update the arrays on the host with the new values got on the device:
!!$acc update host(matr3D, d_w, d_info)
!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr3D, d_w, d_info)
!$acc exit data delete(d_work)


!!! Print check:
print *, "STATUS=success"

print *, d_info

!!! Matrices print:
print *, "Matrix 3D"
do j = 1, num_k
    print *, "Matrix", j
    do c = 1, n 
        print *, matr3D(c,:,j)
    end do
end do

print *, "Eigenvalues"
do a = 1, n
    print *, d_w(a,:)
end do


!! We want a 1dim array with ordered eigenvalues:
do b = 1, n-1
   minim = d_w(b,1)
   do k = b,n
        if (d_w(k,1) .le. minim ) then 
            indexm = k
            minim = d_w(k,1)
        end if
   end do
   d_w(indexm,1) = d_w(b,1) 
   d_w(b,1) = minim
   d_w_ord(b) = minim
end do
d_w_ord(n) = d_w(n,1)

!! We print in a file the eigenvalues:
OPEN(UNIT=20, FILE="matrix_batched.dat", STATUS="REPLACE")
    do j = 1,n
        write(20,*) d_w_ord(j)
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr3D)
DEALLOCATE(d_w)
DEALLOCATE(STATUS_B)
DEALLOCATE(d_info)
DEALLOCATE(d_work) !!! We could allocate d_work only on the device throughb proper CUDA commands, but we allocated it also on the host

DEALLOCATE(d_w_ord)

!!!! We deallocate the handles created:
STATUS = cusolverDnDestroySyevjInfo(syevj_params)
STATUS = cusolverDnDestroy(cusolver_handle)


END PROGRAM script_diag
