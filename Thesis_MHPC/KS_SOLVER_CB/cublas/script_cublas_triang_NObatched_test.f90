!!!!! This is the script for the triangular linear system solver 
!!!!! provided by cuBlas
!!!!! We solve a linear system with the corresponding matrix equal
!!!!! to a triangular matrix:

PROGRAM script_cublas_triang_noB

USE cudafor !!! Cuda Fortran
USE cublas !! We need cublas NVIDIA library!
USE openacc, only: c_devptr, acc_deviceptr !!! We need c_devptr to 
!! manage device pointers!!

IMPLICIT NONE

!!! ATTENTION!: we set a leading dimension too.
INTEGER:: i, n=6, num_k=4, ld=10 ! We have num_k matrices, one for each k-point

!!!! REAL value for the module of complex values:
REAL(8) :: modul


!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS

!!! In this case we define the arrays a and b as contiguous 3D
!!! Arrays ONLY TO ASSIGN THEIR 2D SLICES TO THE ARRAYS OF POINTERS!
COMPLEX(8), ALLOCATABLE:: matr2D_a(:,:), matr2D_b(:,:) ! Arrays 3Dfor storing all the matrices corresponding to the triangular matrix, and to store all the matrices corresponding to the right hand side!


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
ALLOCATE(matr2D_a(ld,n))
ALLOCATE(matr2D_b(ld,n))


!!! Matr3D_a initialization:
matr2D_a = (0.D0, 0.D0)

!!! Matr3D_b initialization:
matr2D_b = (0.D0, 0.D0)

!Loop to populate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
!!$omp do
    !! Case with n = 6 :
    matr2D_a(1,1) = (2.0D0,  1.0D0)
    matr2D_a(2,2) = (3.0D0, -1.0D0)
    matr2D_a(3,3) = (1.5D0,  2.0D0)
    matr2D_a(4,4) = (-4.0D0, 1.0D0)
    matr2D_a(2,1) = (1.0D0,  0.5D0)
    matr2D_a(3,1) = (0.0D0,  2.0D0)
    matr2D_a(3,2) = (-1.0D0, 0.0D0)
    matr2D_a(4,1) = (0.5D0, -1.0D0)
    matr2D_a(4,2) = (3.0D0,  0.0D0)
    matr2D_a(4,3) = (1.0D0,  1.0D0)
    matr2D_a(1,2) = (0.0D0, 0.0D0)
    matr2D_a(1,3) = (0.0D0, 0.0D0)
    matr2D_a(1,4) = (0.0D0, 0.0D0)
    matr2D_a(2,3) = (0.0D0, 0.0D0)
    matr2D_a(2,4) = (0.0D0, 0.0D0)
    matr2D_a(3,4) = (0.0D0, 0.0D0)
    
    matr2D_b(1,1) = (2.0D0, 1.0D0)
    matr2D_b(2,2) = (2.0D0, -1.0D0)
    matr2D_b(3,3) = (3.0D0, 2.0D0)
    matr2D_b(4,4) = (-1.0D0, 1.0D0)
    matr2D_b(1,2) = (2.0D0, -1.0D0)
    matr2D_b(1,3) = (0.0D0, 0.0D0)
    matr2D_b(1,4) = (1.0D0, 0.0D0)
    matr2D_b(2,1) = (0.0D0, 0.0D0)
    matr2D_b(2,3) = (1.0D0, 1.0D0)
    matr2D_b(2,4) = (0.0D0, 0.0D0)
    matr2D_b(3,1) = (1.0D0, 0.0D0)
    matr2D_b(3,2) = (0.0D0, 0.0D0)
    matr2D_b(3,4) = (0.0D0, -1.0D0)
    matr2D_b(4,1) = (7.0D0, 3.0D0)
    matr2D_b(4,2) = (1.0D0, 0.0D0)
    matr2D_b(4,3) = (0.0D0, 1.0D0)

!!$omp end parallel


!! We copy the matrix 3D for testing on single thread from host to gpu:
!! It is important to observe that we copy to the device, for the moment, only the 3D arrays in order to generate a proper device pointer to
!! their memory storage on the device!
!$acc enter data copyin(matr2D_a)
!$acc enter data copyin(matr2D_b)


STATUS = cublasCreate(cublas_handle)
print *, "Status: ", STATUS


!! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
!$acc host_data use_device(matr2D_a, matr2D_b)
STATUS = cublasZtrsm(cublas_handle, CUBLAS_SIDE_LEFT, &
         CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, &
         n, n, alpha, matr2D_a, ld, matr2D_b, ld)
!$acc end host_data

print *, "STATUS:", STATUS

!!!1 WE USE COPYOUT DIRECTIVE BECAUSE IT INCLUDES BOTH UPDATE HOST VARIABLES AND ALSO FREES THE DEVICE POINTERS!
!$acc exit data copyout(matr3D_a, matr3D_b, matr_a, matr_b)


!!! Print check:
print *, "STATUS=success"


!!! Matrices print:
print *, "Matrix 2D"
do c = 1, n 
    print *, matr2D_b(c,:)
end do


!! We print in a file the result of the product:
OPEN(UNIT=20, FILE="matrix_NObatched.dat", STATUS="REPLACE")
    do j = 1,n
        do a =1,n
            modul = abs(matr2D_b(a,j))
            write(20,*) modul
        end do
    end do
CLOSE(20)
!!!!

DEALLOCATE(matr2D_a)
DEALLOCATE(matr2D_b)

!!!! We deallocate the handles created:
STATUS = cublasDestroy(cublas_handle)


END PROGRAM script_cublas_triang_noB
