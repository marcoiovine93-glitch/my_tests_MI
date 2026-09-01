#!/bin/bash
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 #equivalent to the number of threads gpu per block so we have 8x4 = 32, equal to the dimension of a warp!
#SBATCH --gpus-per-node=4
#SBATCH --time=00:12:00
#SBATCH --exclusive
#SBATCH --account=ict26_mhpc_0            
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg

export OMP_NUM_THREADS=1

ml nvhpc/25.11

ml cuda/12.2

# Compile :
rm -rf build
CC=nvc CXX=nvc++ FC=nvfortran cmake -S . -B build \
 -DQE_FFTW_VENDOR=Internal \
 -DQE_ENABLE_OPENMP=ON \
 -DQE_ENABLE_CUDA=ON \
 -DQE_ENABLE_TEST=OFF \
 -DQE_ENABLE_MPI=OFF
 -DNVFORTRAN_CUDA_CC=80
cd build

make cb_davidson -j

#Execution:
cd bin

nvidia-smi

./cb_davidson.x < ../../examples/si2_11_points.in
