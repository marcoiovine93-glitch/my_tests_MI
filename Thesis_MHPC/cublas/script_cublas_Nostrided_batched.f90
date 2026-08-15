PROGRAM script_cublas_nostrided

USE cudafor !!! Cuda Fortran
USE cublas !! We need cublas NVIDIA library!
USE openacc,  only : c_devptr, acc_deviceptr !! We need c-pointers to be managed as device pointers loaded/stored on the host

IMPLICIT NONE

!!! ATTENTION!: we set a leading dimension too.
INTEGER:: i, n=6, num_k=4, ld=10 ! We have num_k matrices, one for each k-point

!!!! REAL value for the module of complex values:
REAL(8) :: modul


!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS


COMPLEX(8), ALLOCATABLE:: matr3D_a(:,:,:), matr3D_b(:,:,:), matr3D_c(:,:,:) ! Arrays 3D for storing all the matrices corresponding to the 
     ! first product factor, all the the matrices corresponding to the 
     ! second product factor, all the the matrices corresponding to the     ! result of the product


!! We define arrays of pointers:
TYPE(c_devptr), ALLOCATABLE :: matr_a(:)
TYPE(c_devptr), ALLOCATABLE :: matr_b(:)
TYPE(c_devptr), ALLOCATABLE :: matr_c(:)


!! Parameters for cuSOLVER Batched API:
TYPE(cublasHandle) :: cublas_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case

!! Variables for printing results:
INTEGER :: c, j, a, b, ki, m, k ! m, n and k are the dimensions needed for cublas call

!! Coefficients for the cublas matrix multiplication:
COMPLEX(8) :: alpha=(1.D0, 0.D0), beta=(0.D0, 0.D0)

!#if defined(_OPENMP)
 ! USE omp_lib, only: omp_get_thread_num
!#endif

!! Allocation on host of the matrix for the test on the single thread!!:
ALLOCATE(matr3D_a(ld,n,num_k))
ALLOCATE(matr3D_b(ld,n,num_k))
ALLOCATE(matr3D_c(ld,n,num_k))


!!We allocate the arrays of pointers:
ALLOCATE(matr_a(num_k))
ALLOCATE(matr_b(num_k))
ALLOCATE(matr_c(num_k))


!!! Matr3D initialization:
matr3D_a = (0.D0, 0.D0)
matr3D_b = (0.D0, 0.D0)
matr3D_c = (0.D0, 0.D0)


!Loop to populate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
do i = 1, num_k
    
    !! Case with n = 6 :
    matr3D_a(1,1,i) = (1.0D0, 1.0D0)
    matr3D_a(2,2,i) = (2.0D0, -1.0D0)
    matr3D_a(3,3,i) = (3.0D0, 2.0D0)
    matr3D_a(4,4,i) = (-1.0D0, 1.0D0)
    matr3D_a(5,5,i) = (4.0D0, -2.0D0)
    matr3D_a(6,6,i) = (5.0D0, 1.0D0)
    matr3D_a(1,2,i) = (2.0D0, -1.0D0)
    matr3D_a(1,3,i) = (0.0D0, 0.0D0)
    matr3D_a(1,4,i) = (0.0D0, 0.0D0)
    matr3D_a(1,5,i) = (1.0D0, 0.0D0)
    matr3D_a(1,6,i) = (0.0D0, -1.0D0)
    matr3D_a(2,1,i) = (0.0D0, 0.0D0)
    matr3D_a(2,3,i) = (1.0D0, 1.0D0)
    matr3D_a(2,4,i) = (0.0D0, 0.0D0)
    matr3D_a(2,5,i) = (0.0D0, 0.0D0)
    matr3D_a(2,6,i) = (2.0D0, 0.0D0)
    matr3D_a(3,1,i) = (1.0D0, 0.0D0)
    matr3D_a(3,2,i) = (0.0D0, 0.0D0)
    matr3D_a(3,4,i) = (0.0D0, -1.0D0)
    matr3D_a(3,5,i) = (0.0D0, 0.0D0)
    matr3D_a(3,6,i) = (0.0D0, 0.0D0)
    matr3D_a(4,1,i) = (0.0D0, 0.0D0)
    matr3D_a(4,2,i) = (1.0D0, 0.0D0)
    matr3D_a(4,3,i) = (0.0D0, 1.0D0)
    matr3D_a(4,5,i) = (2.0D0, 0.0D0)
    matr3D_a(4,6,i) = (0.0D0, 0.0D0)
    matr3D_a(5,1,i) = (2.0D0, 0.0D0)
    matr3D_a(5,2,i) = (0.0D0, 0.0D0)
    matr3D_a(5,3,i) = (0.0D0, 0.0D0)
    matr3D_a(5,4,i) = (0.0D0, -1.0D0)
    matr3D_a(5,6,i) = (1.0D0, 0.0D0)
    matr3D_a(6,1,i) = (0.0D0, 0.0D0)
    matr3D_a(6,2,i) = (1.0D0, 0.0D0)
    matr3D_a(6,3,i) = (2.0D0, 0.0D0)
    matr3D_a(6,4,i) = (0.0D0, 0.0D0)
    matr3D_a(6,5,i) = (1.0D0, 1.0D0)
    !! WE save the current matrix on the thread in the general array
    !general_matr(:,:,i) = matr
    
     matr3D_b(1,1,i) = (2.0D0, 1.0D0)
     matr3D_b(2,2,i) = (2.0D0, -1.0D0)
     matr3D_b(3,3,i) = (3.0D0, 2.0D0)
     matr3D_b(4,4,i) = (-1.0D0, 1.0D0)
     matr3D_b(5,5,i) = (4.0D0, -2.0D0)
     matr3D_b(6,6,i) = (5.0D0, 1.0D0)
     matr3D_b(1,2,i) = (2.0D0, -1.0D0)
     matr3D_b(1,3,i) = (0.0D0, 0.0D0)
     matr3D_b(1,4,i) = (1.0D0, 0.0D0)
     matr3D_b(1,5,i) = (1.0D0, 0.0D0)
     matr3D_b(1,6,i) = (0.0D0, -1.0D0)
     matr3D_b(2,1,i) = (0.0D0, 0.0D0)
     matr3D_b(2,3,i) = (1.0D0, 1.0D0)
     matr3D_b(2,4,i) = (0.0D0, 0.0D0)
     matr3D_b(2,5,i) = (0.0D0, 3.0D0)
     matr3D_b(2,6,i) = (2.0D0, 0.0D0)
     matr3D_b(3,1,i) = (1.0D0, 0.0D0)
     matr3D_b(3,2,i) = (0.0D0, 0.0D0)
     matr3D_b(3,4,i) = (0.0D0, -1.0D0)
     matr3D_b(3,5,i) = (0.0D0, 0.0D0)
     matr3D_b(3,6,i) = (0.0D0, 0.0D0)
     matr3D_b(4,1,i) = (7.0D0, 3.0D0)
     matr3D_b(4,2,i) = (1.0D0, 0.0D0)
     matr3D_b(4,3,i) = (0.0D0, 1.0D0)
     matr3D_b(4,5,i) = (2.0D0, 0.0D0)
     matr3D_b(4,6,i) = (10.0D0, 0.0D0)
     matr3D_b(5,1,i) = (2.0D0, 0.0D0)
     matr3D_b(5,2,i) = (0.0D0, 0.0D0)
     matr3D_b(5,3,i) = (0.0D0, -6.0D0)
     matr3D_b(5,4,i) = (0.0D0, -1.0D0)
     matr3D_b(5,6,i) = (1.0D0, 0.0D0)
     matr3D_b(6,1,i) = (0.0D0, 5.0D0)
     matr3D_b(6,2,i) = (1.0D0, 0.0D0)
     matr3D_b(6,3,i) = (2.0D0, 0.0D0)
     matr3D_b(6,4,i) = (4.0D0, 0.0D0)
     matr3D_b(6,5,i) = (1.0D0, 1.0D0)

end do
!!$omp end parallel


!! We copy the matrix 3D for testing on single thread from host to gpu:
!$acc enter data copyin(matr3D_a)
!$acc enter data copyin(matr3D_b)
!$acc enter data copyin(matr3D_c)


!!! We assign the device pointers to the GPU addresses of the 2D slices
!!! corresponding to each 3D matrix to the pointers of the host array:
do a = 1, num_k
    matr_a(a) = acc_deviceptr(matr3D_a(:,:,a))
    matr_b(a) = acc_deviceptr(matr3D_b(:,:,a))
    matr_c(a) = acc_deviceptr(matr3D_c(:,:,a))
end do

!!Now we can copy the arrays of pointers to the device:
!$acc enter data copyin(matr_a)
!$acc enter data copyin(matr_b)
!$acc enter data copyin(matr_c)


STATUS = cublasCreate(cublas_handle)
print *, "Status: ", STATUS


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr3D_a, matr3D_b, matr3D_c, matr_a, matr_b,matr_c)
STATUS = cublasZgemmBatched(cublas_handle, &
     CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, matr_a, &
     ld, matr_b, ld, beta, matr_c, ld, num_k)
!$acc end host_data

print *, "STATUS:", STATUS

!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr3D_a, matr3D_b, matr3D_c, matr_a, matr_b, matr_c)


!!! Print check:
print *, "STATUS=success"


!!! Matrices print:
print *, "Matrix 3D"
do j = 1, num_k
    print *, "Matrix", j
    do c = 1, n 
        print *, matr3D_c(c,:,j)
    end do
end do


!! We print in a file the result of the product:
OPEN(UNIT=20, FILE="matrix_nostrided_batched.dat", STATUS="REPLACE")
    do j = 1,n
        do k =1,n
            modul = abs(matr3D_c(k,j,1))
            write(20,*) modul
        end do
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr3D_a)
DEALLOCATE(matr3D_b)
DEALLOCATE(matr3D_c)

!!!! We deallocate the handles created:
STATUS = cublasDestroy(cublas_handle)


END PROGRAM script_cublas_nostrided
