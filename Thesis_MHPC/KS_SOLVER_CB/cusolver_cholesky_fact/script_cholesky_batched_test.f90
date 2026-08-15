PROGRAM script_cholesky

USE cudafor !!! Cuda Fortran
USE cusolverDn
USE openacc, ONLY: c_devptr, acc_deviceptr !!! We need c_devptr for managing device pointers AND WE NEED acc_deviceptr for giving in output
! the device pointer of the memory storage where the host object has 
! been mapped through an OpenACC copyin.

IMPLICIT NONE

INTEGER:: i, lda = 10, n=6, num_k=4 ! We have num_k matrices, one for each k-point

!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS


COMPLEX(8), ALLOCATABLE:: matr3D(:,:,:) ! Array 3D for storing all the matrices in one array and make easier to assign them pointers to the array of pointers arr_of_ptr.

!!! We define an array of c_devptr directly on the device :
TYPE(c_devptr), ALLOCATABLE :: arr_of_ptr(:)


!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

! cusolverptr argument for info output related to each input matrix:
INTEGER, ALLOCATABLE :: d_info(:)

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
ALLOCATE(matr3D(lda,n,num_k))

!! Allocation on the host of the array of c_pointers:
ALLOCATE(arr_of_ptr(num_k))


!! We allocate d_info:
ALLOCATE(d_info(num_k))


!!! Matr3D initialization:
matr3D = (0.D0, 0.D0)

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
    matr3D(1,1,i) = (10.0D0, 0.0D0)
    matr3D(2,2,i) = (11.0D0, 0.0D0)
    matr3D(3,3,i) = (12.0D0, 0.0D0)
    matr3D(4,4,i) = (13.0D0, 0.0D0)
    matr3D(5,5,i) = (14.0D0, 0.0D0)
    matr3D(6,6,i) = (15.0D0, 0.0D0)
    matr3D(1,2,i) = (2.0D0, -1.0D0)
    matr3D(1,3,i) = (0.0D0, 0.0D0)
    matr3D(1,4,i) = (0.0D0, 0.0D0)
    matr3D(1,5,i) = (1.0D0, 0.0D0)
    matr3D(1,6,i) = (0.0D0, -1.0D0)
    matr3D(2,1,i) = (2.0D0, 1.0D0)
    matr3D(2,3,i) = (1.0D0, 1.0D0)
    matr3D(2,4,i) = (0.0D0, 0.0D0)
    matr3D(2,5,i) = (0.0D0, 0.0D0)
    matr3D(2,6,i) = (2.0D0, 0.0D0)
    matr3D(3,1,i) = (0.0D0, 0.0D0)
    matr3D(3,2,i) = (1.0D0, -1.0D0)
    matr3D(3,4,i) = (0.0D0, -1.0D0)
    matr3D(3,5,i) = (0.0D0, 0.0D0)
    matr3D(3,6,i) = (0.0D0, 0.0D0)
    matr3D(4,1,i) = (0.0D0, 0.0D0)
    matr3D(4,2,i) = (0.0D0, 0.0D0)
    matr3D(4,3,i) = (0.0D0, 1.0D0)
    matr3D(4,5,i) = (2.0D0, 0.0D0)
    matr3D(4,6,i) = (0.0D0, 0.0D0)
    matr3D(5,1,i) = (1.0D0, 0.0D0)
    matr3D(5,2,i) = (0.0D0, 0.0D0)
    matr3D(5,3,i) = (0.0D0, 0.0D0)
    matr3D(5,4,i) = (2.0D0, 0.0D0)
    matr3D(5,6,i) = (1.0D0, 0.0D0)
    matr3D(6,1,i) = (0.0D0, 1.0D0)
    matr3D(6,2,i) = (2.0D0, 0.0D0)
    matr3D(6,3,i) = (0.0D0, 0.0D0)
    matr3D(6,4,i) = (0.0D0, 0.0D0)
    matr3D(6,5,i) = (1.0D0, 0.0D0)
    !! WE save the current matrix on the thread in the general array

end do
!!$omp end parallel



!! We copy the matrix 3D from host to gpu:
!$acc enter data copyin(matr3D)


!!! WARNING:
!!! It is not necessary to copy scalar variables from host to device!!


!! We copy d_info from host to device:
!$acc enter data copyin(d_info)

!!! We assign to each element of arr_of_ptr the device pointer to each matrix stored in matr3D:
do k = 1, num_k 
    arr_of_ptr(k) = acc_deviceptr(matr3D(:,:,k))
end do 


!! We copy d_w from host to device:
!$acc enter data copyin(arr_of_ptr)


STATUS = cusolverDnCreate(cusolver_handle)
print *, "Status: ", STATUS

!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr3D, arr_of_ptr, d_info)
STATUS = cusolverDnZpotrfBatched(cusolver_handle, &
     CUBLAS_FILL_MODE_LOWER, n, arr_of_ptr, lda, d_info(1), num_k)
!$acc end host_data

!! We update the arrays on the host with the new values got on the device:
!!$acc update host(matr3D, d_w, d_info)
!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr3D, arr_of_ptr, d_info)


!!! Print check:
print *, "STATUS=success"

print *, d_info

!!! Matrices print:
print *, "Matrix 3D"
do b = 1, num_k
    print *, "Matrix", b
    do c = 1, n 
        print *, matr3D(c,:,b)
    end do
end do

!! File output print:
OPEN(UNIT=20, FILE="matrix_batched.dat", STATUS="REPLACE")
    do a = 1,n
        do bi = 1,n
            modul(bi,a) = abs(matr3D(a,bi,1))  
            write(20,*) modul(bi,a)
        end do
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr3D)
DEALLOCATE(d_info)
DEALLOCATE(arr_of_ptr)

!!!! We deallocate the handles created:
STATUS = cusolverDnDestroy(cusolver_handle)


END PROGRAM script_cholesky
