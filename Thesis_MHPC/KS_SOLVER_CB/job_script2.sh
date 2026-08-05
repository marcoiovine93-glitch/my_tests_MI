#!/bin/bash
#SBATCH --job-name=qespresso
#SBATCH --error=errorfile.err
#SBATCH --gres=tmpfs:30G
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=32 #28 #56
#SBATCH --hint=nomultithread
#SBATCH --time=00:35:00               # Time limit hrs:min:sec
#SBATCH -p dcgp_usr_prod
#SBATCH --account=ict26_mhpc

#we removed the nomultithread option

module purge

#Modules loading:
#module load openblas/0.3.26--gcc--12.2.0
#module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-tbb/2021.12.0
module load intel-oneapi-mkl


#Compile :
cd build
rm -rf *
cmake .. \
     -DQE_ENABLE_MPI=ON \
     -DQE_ENABLE_SCALAPACK=ON \

cmake --build .

# We go the bin directory:
cd bin

#Run:
mpirun -np 8 ./cb_davidson_main.x -ndiag 1 -nk 4 -i ../../si2_11_points.in
