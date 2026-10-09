## Progress Meeting MHPC & NVIDIA Meeting

### General data:
- NVHPC SDK version
    - Spark: 26.3
    - Leonardo: 25.3
- CUDA version for Leonardo drivers: 12.2
- We didn't use OpenACC device resident arrays in order to guarantee as much as possible compatibility with both
  distributed memory (Leonardo) and unified memory (Spark) architectures


### Objectives








### Strategy

#### Diagonalizer batching

- NVIDIA API : cusolverDnZheevjBatched
    - It needed the declaration of 3D ARRAYS, each slice corresponding to the 2D original reduced matrices
    - In order to get the right numerical results, we set a number of columns (nvec) equal to the leading dimension of the matrices included in the 
      batch:
            - if we set them differently, we observe wrong results (NaN values)
    - The 2D matrices across the threads show different dimension --> we introduced a Padding that preserves convergence and convergence speed
      within the diagonalization


- Declaration of 3D shared arrays across the threads for the kernel call before the iterative part
    - they have been declared as host standard arrays and then copied to the device with OpenACC

- Declaration of 3D shared arrays across the threads for the kernel call inside the iterative part
    - they have been declared as host ALLOCATABLE heap arrays and then copied to the device with OpenACC
    - we defined them as ALLOCATABLE inside the cegterg subroutine because we need to REALLOCATE THEM based on the active threads inside the 
      iterative part of the Davidson diagonalization
        - defining these arrays as allocatable in cb_davidson_main and passing them to the subroutine cegterg subroutine, made necessary to
          DEFINE THE cegterg.f90 as a MODULE in order to have an explicit interface for the subroutine


- Use of Cholevsky decomposition (cusolverDnZpotrfBatched), triangular matrix multiplication (cublasZtrsmBatched) together with the cuSolver diagonalizer   (cusolverDnZheevjBatched) in order to take into account the Overlap matrix
    - We defined all the operations inside the subroutine laxlib_cdiaghg_gpu_batched
    - Cholevsky decomposition:
        - ARRAYS OF POINTERS TO MATRICES were needed
        - we assigned the arrays of pointers to the matrices with the use of the OpenACC directive: c_devloc
        - Sequence of operations:
            1) Cholesky factorization
            2) pre and post multiplication of the Hamiltonian with a triangular matrix multiplication cublasZtrsmBatched (Ly = H and y = wL)
            3) call to cusolverDnZheevjBatched
            4) to retrieve the correct eigenvectors, we do again a triangular matrix multiplication with cublasZtrsmBatched
    

- For the Cholevsky decomposition and the triangular matrix multiplication, we needed to create and initialize a cublas Handle for each OpenMP thread (Open ACC queue)
    - WE APPLIED a similar logic already implemented for the cuSolver handles:
        - we initialized and finalized the cublas Handles in the main program and we reassigned it to the corresponding OpenACC queues inside the 
          subroutine cdiaghg_gpu_batched in the file cdiaghg.f90
        - we set the cublas Handles with the SAVE attribute and we didn't destroy them inside the cdiaghg_gpu_batched subroutine
          --> destrying it for each call to the subroutine caused the results to be incorrect (equal to NaN) even with a serial execution (one OpenMP thread)


##### Synchronization:

- We synchronize by performing:
    - omp barrier and acc wait(async_id) to synchronize both CPU and GPU
    - launch of the Batched API Kernels from a single OpenMP thread --> use of omp single
    - copy back of the 3D arrays into the 2D original ones



#### Zgemm batching

- Initial part before the iterations:
    - we used the API: cublasZgemmStridedBatched
    - REASONS:
        - we have memory contiguity
        - the matrices show the same dimensions and we take the same slices for all of them
    - It was necessary to declare 3D shared arrays containing the 2D matrices corresponding to each thread

- Iterative part:
    - we use the API : cublasZgemmBatched
    - REASONS:
        - the matrices show variable slices that are not constant across the OpenMP threads
    - It was necessary to define shared arrays of pointers to matrices, assigned through the OpenACC directives acc_deviceptr and c_devloc


## Next steps

