#!/bin/bash
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32 #equivalent to the number of threads gpu per block so we have 8x4 = 32, equal to the dimension of a warp!
#SBATCH --gres=gpu:1
#SBATCH --time=00:12:00
#SBATCH --exclusive
#SBATCH --account=ict26_mhpc_0            
#SBATCH --partition=boost_usr_prod

export OMP_NUM_THREADS=1
#export OMP_THREADS=1

# Modules load accelerator error:
module purge
module load nvhpc/25.11
#module load hpcx-mpi
module load hpcx-mpi/2.25.1
module load glibc/2.28--gcc--12.2.0-gi6mmti
#module load glibc
module load fftw/3.3.10--hpcx-mpi--2.25.1--nvhpc--25.11
#module load glibc
#module load fftw


# Open MP error:
#module purge
#module load nvhpc/25.11
#module load hpcx-mpi
#module load hpcx-mpi/2.25.1
#module load glibc/2.28--gcc--12.2.0-gi6mmti
#module load glibc
#module load fftw/3.3.10--hpcx-mpi--2.25.1--nvhpc--25.11
#module load glibc
#module load fftw



# NVHPC 24.5 :
#module purge
#module load nvhpc/24.5
#module load glibc/2.28--gcc--12.2.0-gi6mmti hpcx-mpi/2.19
#module load fftw/3.3.10--hpcx-mpi--2.19--nvhpc--24.5


export FFTW3_ROOT="$FFTW_HOME"
export FFTW_ROOT="$FFTW_HOME"
export FFTW_DIR="$FFTW_HOME"


# Compile :
#rm -rf build
#CC=nvc CXX=nvc++ FC=nvfortran cmake -S . -B build \
 # -DCMAKE_Fortran_FLAGS="-acc -cuda -cudalib=cusolver -Mpreprocess -D__CUDA -gpu=cc80" \
 # -DQE_FFTW_VENDOR=FFTW3 \
 # -DFFTW3_ROOT="$FFTW3_ROOT" \
 # -DCMAKE_PREFIX_PATH="$FFTW3_ROOT" \
           #-DFFTW3_ROOT="$FFTW_HOME" \
           #-DCMAKE_PREFIX_PATH="$FFTW_HOME" \
 # -DBLA_VENDOR=Generic \
 # -DQE_ENABLE_OPENMP=ON \
 # -DQE_ENABLE_CUDA=ON \
 # -DQE_ENABLE_TEST=OFF \
 # -DQE_ENABLE_MPI=OFF

cd build
#make cb_davidson -j

#nvidia-smi

#make qe_utilxlib qe_fftxlib -j

make cb_davidson -j

#strings bin/cb_davidson.x | grep -iE "cuMemAlloc|cuLaunchKernel|cudaMalloc" | head -10

#Execution:
cd bin
#nvaccelinfo -v

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo $NVHPC_CUDA_HOME
echo $CUDA_HOME
echo $CUDA_PATH

nvaccelinfo
which nvfortran
nvfortran --version
nvfortran -V

nvidia-smi

srun --gres=gpu:1 cb_davidson.x < ../../examples/si2_11_points.in


#make VERBOSE=1 cb_davidson -j32 2>&1 | tee build_verbose.log
#grep "nvfortran" build_verbose.log | grep -o '\-acc\|\-cuda\|\-gpu=cc80' | sort | uniq -c
#ls -la bin/cb_davidson.x
