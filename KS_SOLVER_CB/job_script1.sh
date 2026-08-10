#!/bin/bash
#SBATCH --error=job_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 #equivalent to the number of threads gpu per block so we have 8x4 = 32, equal to the dimension of a warp!
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --exclusive
#SBATCH --account=ict26_mhpc_0            
#SBATCH --partition=boost_usr_prod

module purge

#export OMP_THREADS=1


#Modules loading:
#module load openblas/0.3.26--gcc--12.2.0
#module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2


module load nvhpc/24.5
module load hpcx-mpi
#module load cuda/12.2
#module load cuda/12.2


module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-tbb/2021.12.0
module load intel-oneapi-mkl



# -L/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/math_libs/lib64/ -lcublas

#Compile:
#mpicxx -I$OPENBLAS_INC -I./include -L$OPENBLAS_LIB -o matrix.x src/main.cpp -lopenblas

#Compile:
#mpic++ -I./include -L/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/math_libs/lib64/ -o matrix.x src/main.cpp -lcublas -lcudart


#nvc++ -o exe -I./include src/main.cpp -acc -gpu=cc80,cuda12.4 -Minfo=acc -lcudart -lnvToolsExt

#nvc++ \
 # -I./include \
 # -I$HPCX_MPI_DIR/include \
  #src/main.cpp \
  #-o exe \
  #-acc \
  #-gpu=cc80 \
  #-Minfo=acc \
  #-L$HPCX_MPI_DIR/lib \
  #-lmpi


#mpic++ -I./include -I$CUDA_INC -L/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/math_libs/lib64/ -o matrix.x src/main.cpp -lcublas -lcudart


# Compile: 
nvfortran -acc -cuda -cudalib=cusolver -o script_diag script_cusolver_batched_test.f90 


#Run:
srun ./script_diag
