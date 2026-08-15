!!!!! This is the script for the triangular linear system solver 
!!!!! provided by cuBlas
!!!!! We solve a linear system with the corresponding matrix equal
!!!!! to a triangular matrix:

PROGRAM script_cublas_triang

USE cudafor !!! Cuda Fortran
USE cublas !! We need cublas NVIDIA library!
USE openacc, only: c_devptr, acc_deviceptr !!! We need c_devptr to 
!! manage device pointers!!

IMPLICIT NONE

!!! ATTENTION!: we set a leading dimension too.
INTEGER:: i, n=4, num_k=4, ld=10 ! We have num_k matrices, one for each k-point

!!!! REAL value for the module of complex values:
REAL(8) :: modul


!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS

!!! In this case we define the arrays a and b as contiguous 3D
!!! Arrays ONLY TO ASSIGN THEIR 2D SLICES TO THE ARRAYS OF POINTERS!
COMPLEX(8), ALLOCATABLE:: matr3D_a(:,:,:), matr3D_b(:,:,:) ! Arrays 3Dfor storing all the matrices corresponding to the triangular matrix, and to store all the matrices corresponding to the right hand side!


!! We define 2 arrays of pointers of the type provided by the OpenACC
!! module c_devptr --> THEY ARE LIKE C-POINTERS!!
TYPE(c_devptr), ALLOCATABLE :: matr_a(:)
TYPE(c_devptr), ALLOCATABLE :: matr_b(:)


!! Parameters for cuSOLVER Batched API:
TYPE(cublasHandle) :: cublas_handle !! Single handle for the test of single thread/stream --> sichronous execution
!TYPE(cusolverDnHandle), ALLOCATABLE :: cusolver_handle(:) !! Multi thread asynchronous case


!! Variables for printing results:
INTEGER :: c, j, a, b, ki, m, k ! m, n and k are the dimensions needed for cublas call

!! Coefficients for the cublas matrix multiplication:
COMPLEX(8) :: alpha=(1.D0, 0.D0)

!#if defined(_OPENMP)
 ! USE omp_lib, only: omp_get_thread_num
!#endif

!! Allocation on host of the matrix for the test on the single thread!!:
ALLOCATE(matr3D_a(ld,n,num_k))
ALLOCATE(matr3D_b(ld,n,num_k))


!!! Allocation of array of pointers:
ALLOCATE(matr_a(num_k))
ALLOCATE(matr_b(num_k))


!!! Matr3D_a initialization:
matr3D_a = (0.D0, 0.D0)

!!! Matr3D_b initialization:
matr3D_b = (0.D0, 0.D0)

!Loop to populate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
do i = 1, num_k
    !! Case with n = 6 :
    matr3D_a(1,1,i) = (2.0D0,  1.0D0)
    matr3D_a(2,2,i) = (3.0D0, -1.0D0)
    matr3D_a(3,3,i) = (1.5D0,  2.0D0)
    matr3D_a(4,4,i) = (-4.0D0, 1.0D0)
    matr3D_a(2,1,i) = (1.0D0,  0.5D0)
    matr3D_a(3,1,i) = (0.0D0,  2.0D0)
    matr3D_a(3,2,i) = (-1.0D0, 0.0D0)
    matr3D_a(4,1,i) = (0.5D0, -1.0D0)
    matr3D_a(4,2,i) = (3.0D0,  0.0D0)
    matr3D_a(4,3,i) = (1.0D0,  1.0D0)
    matr3D_a(1,2,i) = (0.0D0, 0.0D0)
    matr3D_a(1,3,i) = (0.0D0, 0.0D0)
    matr3D_a(1,4,i) = (0.0D0, 0.0D0)
    matr3D_a(2,3,i) = (0.0D0, 0.0D0)
    matr3D_a(2,4,i) = (0.0D0, 0.0D0)
    matr3D_a(3,4,i) = (0.0D0, 0.0D0)
    
    matr3D_b(1,1,i) = (2.0D0, 1.0D0)
    matr3D_b(2,2,i) = (2.0D0, -1.0D0)
    matr3D_b(3,3,i) = (3.0D0, 2.0D0)
    matr3D_b(4,4,i) = (-1.0D0, 1.0D0)
    matr3D_b(1,2,i) = (2.0D0, -1.0D0)
    matr3D_b(1,3,i) = (0.0D0, 0.0D0)
    matr3D_b(1,4,i) = (1.0D0, 0.0D0)
    matr3D_b(2,1,i) = (0.0D0, 0.0D0)
    matr3D_b(2,3,i) = (1.0D0, 1.0D0)
    matr3D_b(2,4,i) = (0.0D0, 0.0D0)
    matr3D_b(3,1,i) = (1.0D0, 0.0D0)
    matr3D_b(3,2,i) = (0.0D0, 0.0D0)
    matr3D_b(3,4,i) = (0.0D0, -1.0D0)
    matr3D_b(4,1,i) = (7.0D0, 3.0D0)
    matr3D_b(4,2,i) = (1.0D0, 0.0D0)
    matr3D_b(4,3,i) = (0.0D0, 1.0D0)


end do
!!$omp end parallel


!! We copy the matrix 3D for testing on single thread from host to gpu:
!! It is important to observe that we copy to the device, for the moment, only the 3D arrays in order to generate a proper device pointer to
!! their memory storage on the device!
!$acc enter data copyin(matr3D_a)
!$acc enter data copyin(matr3D_b)


!!! We assign to each c-pointer of the array of pointers, the device
!!! pointer associated to each 2D slice of the 3D contiguous matrices:
do k = 1, num_k 
   matr_a(k) = acc_deviceptr(matr3D_a(:,:,k))
   matr_b(k) = acc_deviceptr(matr3D_b(:,:,k))
end do

!!! Now we can copyin matr_a and matr_b to the device:
!$acc enter data copyin(matr_a)
!$acc enter data copyin(matr_b)



STATUS = cublasCreate(cublas_handle)
print *, "Status: ", STATUS


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr3D_a, matr3D_b, matr_a, matr_b)
STATUS = cublasZtrsmBatched(cublas_handle, CUBLAS_SIDE_LEFT, &
         CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, &
         n, n, alpha, matr_a, ld, matr_b, ld, num_k)
!$acc end host_data

print *, "STATUS:", STATUS

!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr3D_a, matr3D_b, matr_a, matr_b)


!!! Print check:
print *, "STATUS=success"


!!! Matrices print:
print *, "Matrix 3D"
do j = 1, num_k
    print *, "Matrix", j
    do c = 1, n 
        print *, matr3D_b(c,:,j)
    end do
end do


!! We print in a file the result of the product:
OPEN(UNIT=20, FILE="matrix_batched.dat", STATUS="REPLACE")
    do j = 1,n
        do a =1,n
            modul = abs(matr3D_b(a,j,1))
            write(20,*) modul
        end do
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr3D_a)
DEALLOCATE(matr3D_b)
DEALLOCATE(matr_a)
DEALLOCATE(matr_b)

!!!! We deallocate the handles created:
STATUS = cublasDestroy(cublas_handle)


END PROGRAM script_cublas_triang
