PROGRAM script_cholesky

USE cudafor !!! Cuda Fortran
USE cusolverDn
USE openacc !!! We need c_devptr for managing device pointers AND WE NEED acc_deviceptr for giving in output
! the device pointer of the memory storage where the host object has 
! been mapped through an OpenACC copyin.

IMPLICIT NONE

INTEGER:: i, lda = 10, n=6 ! We have num_k matrices, one for each k-point

!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS, lwork


!!! For the not batched API, we need to define a workspace of size lwork :
COMPLEX(8), ALLOCATABLE :: d_work(:)


COMPLEX(8), ALLOCATABLE:: matr2D(:,:) ! Array 3D for storing all the matrices in one array and make easier to assign them pointers to the array of pointers arr_of_ptr.


!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

! cusolverptr argument for info output related to each input matrix:
INTEGER :: d_info

!! Variables for printing results:
INTEGER :: c, j, a, b, k, bi

!! Matrix with the modules of the complex values:
REAL(8), ALLOCATABLE :: modul(:,:)

!#if defined(_OPENMP)
 ! USE omp_lib, only: omp_get_thread_num
!#endif

!!Allocation of the modules of complex values matrix:
ALLOCATE(modul(n,n))

!! Allocation on host of the 3d matrix:
ALLOCATE(matr2D(lda,n))


!!! Matr2D initialization:
matr2D = (0.D0, 0.D0)

!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
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
    matr2D(1,1) = (10.0D0, 0.0D0)
    matr2D(2,2) = (11.0D0, 0.0D0)
    matr2D(3,3) = (12.0D0, 0.0D0)
    matr2D(4,4) = (13.0D0, 0.0D0)
    matr2D(5,5) = (14.0D0, 0.0D0)
    matr2D(6,6) = (15.0D0, 0.0D0)
    matr2D(1,2) = (2.0D0, -1.0D0)
    matr2D(1,3) = (0.0D0, 0.0D0)
    matr2D(1,4) = (0.0D0, 0.0D0)
    matr2D(1,5) = (1.0D0, 0.0D0)
    matr2D(1,6) = (0.0D0, -1.0D0)
    matr2D(2,1) = (2.0D0, 1.0D0)
    matr2D(2,3) = (1.0D0, 1.0D0)
    matr2D(2,4) = (0.0D0, 0.0D0)
    matr2D(2,5) = (0.0D0, 0.0D0)
    matr2D(2,6) = (2.0D0, 0.0D0)
    matr2D(3,1) = (0.0D0, 0.0D0)
    matr2D(3,2) = (1.0D0, -1.0D0)
    matr2D(3,4) = (0.0D0, -1.0D0)
    matr2D(3,5) = (0.0D0, 0.0D0)
    matr2D(3,6) = (0.0D0, 0.0D0)
    matr2D(4,1) = (0.0D0, 0.0D0)
    matr2D(4,2) = (0.0D0, 0.0D0)
    matr2D(4,3) = (0.0D0, 1.0D0)
    matr2D(4,5) = (2.0D0, 0.0D0)
    matr2D(4,6) = (0.0D0, 0.0D0)
    matr2D(5,1) = (1.0D0, 0.0D0)
    matr2D(5,2) = (0.0D0, 0.0D0)
    matr2D(5,3) = (0.0D0, 0.0D0)
    matr2D(5,4) = (2.0D0, 0.0D0)
    matr2D(5,6) = (1.0D0, 0.0D0)
    matr2D(6,1) = (0.0D0, 1.0D0)
    matr2D(6,2) = (2.0D0, 0.0D0)
    matr2D(6,3) = (0.0D0, 0.0D0)
    matr2D(6,4) = (0.0D0, 0.0D0)
    matr2D(6,5) = (1.0D0, 0.0D0)
    !! WE save the current matrix on the thread in the general array

!!$omp end parallel



!! We copy the matrix 3D from host to gpu:
!$acc enter data copyin(matr2D)


!!! WARNING:
!!! It is not necessary to copy scalar variables from host to device!!


!! We copy d_info from host to device:
!$acc enter data copyin(d_info)


STATUS = cusolverDnCreate(cusolver_handle)
print *, "Status: ", STATUS


!! We get the required buffer size:
!$acc host_data use_device(matr2D)
STATUS = cusolverDnZpotrf_bufferSize(cusolver_handle, &
                                     CUBLAS_FILL_MODE_LOWER, &
                                     n, matr2D, lda, lwork);
!$acc end host_data


!! We allocate d_work on the host:
ALLOCATE(d_work(lwork))

!! We need to create a storage on the device for d_work:
!$acc enter data create(d_work)

!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr2D, d_work)
STATUS = cusolverDnZpotrf(cusolver_handle, &
                                     CUBLAS_FILL_MODE_LOWER, &
                                     n, matr2D, lda, d_work, lwork, &
                                     d_info);
!$acc end host_data

!! We update the arrays on the host with the new values got on the device:
!!$acc update host(matr3D, d_w, d_info)
!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr2D)
!$acc exit data delete(d_work)

!!! Print check:
print *, "STATUS=success"

!!! Matrices print:
print *, "Matrix 2D"
do c = 1, n 
        print *, matr2D(c,:)
end do

!! File output print:
OPEN(UNIT=20, FILE="matrix.dat", STATUS="REPLACE")
    do a = 1,n
        do bi = 1,n
            modul(bi,a) = abs(matr2D(a,bi))  
            write(20,*) modul(bi,a)
        end do
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr2D)
DEALLOCATE(d_work)

!!!! We deallocate the handles created:
STATUS = cusolverDnDestroy(cusolver_handle)


END PROGRAM script_cholesky
