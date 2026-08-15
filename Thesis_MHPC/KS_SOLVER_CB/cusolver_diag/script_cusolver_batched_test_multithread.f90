PROGRAM script_diag

USE cudafor !!! Cuda Fortran
USE cusolverDn
USE openacc
USE iso_c_binding, ONLY: c_ptr, c_null_ptr !! We need the binding to pass C-pointers and the C-pointer to null --> IN THIS WAY WE CAN AVOID TOUSE DERIVED TYPE for the info parameter of cudasolverDn

#if defined(_OPENMP)
     USE omp_lib, only: omp_get_thread_num, omp_set_num_threads, omp_get_num_threads
#endif

IMPLICIT NONE


!! Variable to store the output of the createDn (success or not)
INTEGER :: STATUS

!! We define the number of threads:
INTEGER :: n_threads
!! We set the cuda streams:
!!INTEGER(cuda_stream_kind) :: stream


! In this case I use an allocatable, not a pointer!!
COMPLEX(8), ALLOCATABLE :: general_matr(:,:,:) ! Array 3D for multistream case

COMPLEX(8), ALLOCATABLE :: matr(:,:) ! Array 2D for the single threads

!COMPLEX(8), ALLOCATABLE:: matr3D(:,:,:) ! Array 3D for storing all the matrices in one matrix for testing the CuSolvr Batched for 1 or multiple threads/streams!!

REAL(8), ALLOCATABLE :: d_w(:,:) ! Stores the eigenvalues of every array matrix



!!!! WE MOVE THE FOLLOWING LINE AT THE BEGINNING, SO IN THIS WE CAN DEFINE THE ARRAYS SIZE AND THE NUMBER OF ARRAYS AT THE COMPILE TIME AND SAVE 
!!!! EXECUTION TIME!!!!
INTEGER :: i, n=3, num_k=4 ! We have num_k matrices, one for each k-point


!! Parameters for cuSOLVER Batched API:
TYPE(cusolverDnHandle) :: cusolver_handle !! Array handle for the test of multi thread/stream --> asynchronous execution
!! WARNING: WE don't have a handle for each thread, because we want
!! the Batched API kernel to be called by one single thread!!!
!! we have a handle for each thread!!!
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
INTEGER :: c, j, a, b


n_threads = num_k

!!! Allocation of cusolverHandle :
!ALLOCATE(cusolver_handle(n_threads))


! Allocation on the host
ALLOCATE(general_matr(n, n, num_k))

!! Allocation on host of the matrix for the test on the single thread!!:
!ALLOCATE(matr3D(n,n,num_k))


!! We allocate d_info:
ALLOCATE(d_info(num_k))

!! We allocate d_w :
ALLOCATE(d_w(n,num_k))


!!! Matr3D initialization:
!matr3D = (0.D0, 0.D0)

!!! Eigenvalue matrix initialization:
d_w = (0.D0)


!! We set the number of threads:
#if defined(_OPENMP)
     call omp_set_num_threads(num_k)
#endif


!Loop to poulate the big array for each thread:
!! We make the matr as private and accessible by one thread to make distinction 
!$omp parallel num_threads(num_k) private(i, matr) shared(general_matr)
    !$omp do
    do i = 1, num_k
        print *, "Current thread number: ", omp_get_thread_num()
        ALLOCATE(matr(n,n))
        matr(1,1) = (2.0D0, 0.0D0)
        matr(2,2) = (2.0D0, 0.0D0)
        matr(3,3) = (2.0D0, 0.0D0)
        matr(1,2) = (0.0D0, 0.0D0)
        matr(1,3) = (0.0D0, -1.0D0)
        matr(2,3) = (0.0D0, 0.0D0)
        matr(3,1) = (0.0D0, 1.0D0)
        matr(3,2) = (0.0D0, 0.0D0)
        matr(2,1) = (0.0D0, 0.0D0)

        !! WE save the current matrix on the thread in the general array
        general_matr(:,:,i) = matr
    
        !! We need to deallocate inside the loop!!
        DEALLOCATE(matr)

    end do
    
    !!!! Let's remember that the end do subtedly runs a omp barrier!!!
    !$omp end do

    !! We want that the next part of the omp parallel to be performed only by the first thread available after the barrier implicit in the end do
    !$omp master
    
    !! We copy the general matrix on the GPU:
    !$acc enter data copyin(general_matr)


    !! We copy the matrix 3D for testing on single thread from host to gpu:
    !!$acc enter data copyin(matr3D)

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
    !$acc host_data use_device(general_matr, d_w)
    STATUS = cusolverDnZheevjBatched_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, general_matr, n, d_w, lwork, syevj_params, num_k)
    !$acc end host_data

    print *, "bufferSize STATUS =", STATUS, " lwork =", lwork


    !!! We allocate d_work after retrieved the Buffersize with the previousfunction:
    ALLOCATE(d_work(lwork))

    !! We copy d_work from host to device:
    !!$acc enter data copyin(d_work)
    !$acc enter data create(d_work)


    !! With OpenACC we pass directly the real addresses on the GPU --> we need to pass data that is already on the GPU!!!!
    !$acc host_data use_device(general_matr, d_w, d_work, d_info)
    STATUS = cusolverDnZheevjBatched(cusolver_handle, &
    CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, &
    n, general_matr, n, d_w, d_work, lwork, d_info(1), syevj_params, num_k)
    !$acc end host_data

    !! We update the arrays on the host with the new values got on the device and we deallocte the device arrays:
    !$acc exit data copyout(general_matr, d_w, d_info)
    !$acc exit data delete(d_work) 
    
    !$omp end master
    
    !!! We run an explicit barrier because we need to make sure that all the threads got the new host variables values computed by the master thread!!:
    !$omp barrier

!! We end the omp parallel region:
!$omp end parallel

!!! Print check:
print *, "STATUS=success"

print *, d_info

!!! Matrices print:
print *, "General matrix"
do j = 1, num_k
    print *, "Matrix", j
    do c = 1, n 
        print *, general_matr(c,:,j)
    end do
end do

print *, "Eigenvalues"
do a = 1, n
    print *, d_w(a,:)
end do


DEALLOCATE(general_matr)
DEALLOCATE(d_w)
DEALLOCATE(d_info)
DEALLOCATE(d_work)

!!!! We deallocate the handles created:
STATUS = cusolverDnDestroySyevjInfo(syevj_params)

STATUS = cusolverDnDestroy(cusolver_handle)

END PROGRAM script_diag
