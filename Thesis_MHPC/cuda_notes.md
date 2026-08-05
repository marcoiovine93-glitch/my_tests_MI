# Compute Unified Device Architecture (CUDA)


## How to use CUDA on an HPC cluster

To use the GPUs on an HPC cluster you typically need to load the NVIDIA HPC SDK
module. This module contains, among many other things, the NVIDIA compilers that
can translate the CUDA APIs in instructions executable by the GPU.
On Leonardo you can load the module via:
```module load nvhpc```
This will load the default version of the SDK present on the cluster (which, at
the time of writing, is the 24.5). In order to check that the module loaded
successfully you can run the command:
```which nvcc```
It should output the path to the nvcc compiler.
To compile a cuda source code, use the aforementioned nvcc compiler:
```nvcc -o myexe mycode.cu -arch=sm_80```
The last option is to specify the compute capabilities of the GPUs on the
cluster: for Leonardo it is 80.

### How to require GPUs in the slurm jobfile

To have access to GPUs on Leonardo you need to use the Boost partition (the one
equipped with GPUs) and require them in the jobfile as follows (note that the
option ```--gres=gpu:``` selects the amount of GPUs **per node**)
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --account="your_account_name"
#SBATCH --partition=boost_usr_prod


module purge
module load nvhpc

./myexe

```

## First steps in CUDA

As discussed in the slides, the basic "workflow" of an heterogeneous code is the
following:

- Copy data from the host (CPU) to the device  (GPU)

- Process the data on the GPU

- Copy the result back onto the host

So we need to understand how to do three things:

- Manage memory allocation on the GPU to accommodate the data coming from the
host

- Copy the data back and forth between host and device

- Write functions that can process the data on the device taking advantage of
the massive parallelism offered by the hardware

#### Headers and modules

In order to use the CUDA API one needs to include the proper headers in C/C++
and modules in Fortran:

```c
#include <cuda_runtime.h>
```
or

```f90
use cudafor
```

Other modules or headers might be necessary when using specific libraries: see
the proper documentation.
In this notes, the focus will be mainly on the C/C++ API, but the Fortran
one is basically the same.

### Memory allocation

To manage memory on the device, CUDA provides the `cudaMalloc` and `cudaFree`
APIs, which mimic the behaviour of the C/C++ `malloc` and `free`:

```c
int * device_data;
size_t n = 100;
size_t size_bytes = n*sizeof(int);

cudaMalloc(&device_data,size_bytes);

cudaFree(device_data);
```

> [!NOTE]
> Note that the cudaMalloc API takes a double pointer as an argument (why?)

### Data management

To copy data between host and device CUDA provides the `cudaMemcpy` API:

```c
int * device_data, *host_data;
size_t n = 100;
size_t size_bytes = n*sizeof(int);

host_data = (int *) malloc(size_bytes);
cudaMalloc(device_data,size_bytes);

cudaMemcpy(device_data,host_data,cudaMemcpyHostToDevice);

// data processing on GPU

cudaMemcpy(host_data,device_data,cudaMemcpyDeviceToHost);

print(data)

cudaFree(device_data); free(host_data);
```

Notice that, since we need both the host and the device counterpart of each
buffer, there is an intrinsic "doubling" of the variables. See the Unified
Memory section for an alternative way of handling memory that leaves most of the
effort to the CUDA runtime.

#### Unified Memory

Unified Memory, introduced in CUDA 6.0, provides a single address space for host
and device. This allows to allocate memory accessible from both CPU and GPU,
meaning that the runtime will take care of performing memory copies when needed.
To use this feature, one must allocate memory using the `cudaMallocManaged` API.
For instance, the counterpart of the snippet above would be:

```c
int *data;
size_t n = 100;
size_t size_bytes = n*sizeof(int);


cudaMallocManaged(data,size_bytes);

// data processing on GPU

print(data);
cudaFree(data);
```
Using the managed memory simplifies the code, lifting much
of the complexity of memory management from the programmer hands. On the other
hand, it entails relinquishing fine control over data movement to the runtime.
The latter can be helped using `cudaMemAdvise` and `cudaPrefetchAsynch`: more
informations can be found on the CUDA API documentation.

### Compute partitioning

CUDA allows for three different types of functions, according to their calling
and execution location:
|  | \_\_host__ | \_\_global\_\_ |\_\_device\_\_ |
| --------------- | --------------- | --------------- | --------------- |
| Called          | host            | host              | device |
| Executed        | host            | device            | device |

Global functions are also called **kernels**.
The majority of the coding effort in a CUDA code is typically devoted to write efficient
kernels. As will be evident in a while (and also in subsequent courses in the
master) there are a lot of moving parts that one needs to keep under control and
writing functions that fully exploit the GPU capabilities is far from trivial.
The first thing to understand now is how to utilize the threads that the device
can spawn.

#### Trivial example

Before delving into the topic of thread mapping, though,  let's take a look to a very simple
CUDA code that sums two integers on the GPU. Of course, to perform this
operation on the GPU makes little sense (why?), but the snippet illustrates most
of the things that have been explained until now.

```c
__global__ void sum (int a, int b, int *sum) {

*sum=a+b;

}

int main(){

  int *dev_sum, h_sum;

  cudaMalloc(&dev_sum, sizeof(int));

  sum<<<1,1>>>(1,2,dev_sum);

  cudaMemcpy(&h_sum, dev_sum, cudaMemcpyDeviceToHost);

  printf(“%d\n”, h_sum);

  cudaFree(dev_sum);

return 0;

```

Notice the `__global__` prefix in front of the kernel definition.
CUDA kernels are all `void` functions, so one needs to pass the pointer to
the memory where the result will be stored. Disregard for now the funny brackets
syntax at kernel launch: in this case the numbers mean that we are launching
this kernel using only one thread since we are performing a serial operation.



> [!IMPORTANT]
> Kernel calls are **asynchronous**: they return immediately to the host. On the
> other hand, memory allocations and data movements are **synchronous** (at least the ones that we will se in this course): the
> host is blocked until all the device work is finished. If you want the CPU to
> wait for the kernel to finish, you can use `cudaDeviceSynchronize()` after
> the kernel call


## Thread management

### Grids and blocks

Recall that, as discussed in the slides, threads are organized in blocks which
are in turn organized in a grid.
When a kernel is launched, one has to use the so-called chevron notation to
specify the number of blocks in the grid and the number of threads in each
block. The syntax is as follows:
`mykernel<<<grid_size, block_size>>> (args)`

Where `grid_size` specifies the (x,y,z) dimensions of the grid (in units of
blocks) and `block_size` specifies the (x,y,z) dimensions of each block (in
units of threads). CUDA provides a structure to specify these sizes, called
`Dim3`:

```
Dim3 grid_size  = (4,5,6);
Dim3 block_size = (10,11,12);
```

In this way, for example, the kernel would be launched on a grid with edge of 4
blocks on the x axis, 5 blocks on the y axis and 6 blocks on the z axis, for a
total of 4x5x6 blocks.
Furthermore, each block would have an edge of 10,11,12 threads on the x,y,z
direction for a total of 10x11x12 threads per block. Each one of these threads
will then execute the instructions written in the kernel.

> [!note]
> The first number  (the x direction) is always required, but the y and z
components are facultative and will be set to 1 if not specified. It is also
possible to pass directly integers in the chevron brackets: this is equivalent
to spawn a one dimensional grid/block. In the example above (the sum of two
integers), the kernel was launched on a grid of one block composed by one
thread, i.e. the runtime spawned only one thread.

### Thread mapping

As mentioned during the lecture, GPUs are mainly geared for data parallelism,
i.e. the input buffer should be split as evenly as possible between the threads who
should  then perform the same operation on their chunk of data.
The question now is how to map a buffer onto the threads. As discussed above, in general,
both threads and blocks are structured in a three-dimensional geometric structure:
the threads in a block and the blocks in a grid.
Therefore, each block will be identified by three indexes, representing
its (x,y,z) **global coordinates in the full three-dimensional grid**. Similarly, each
thread will be identified three integers, representing its (x,y,z) **local coordinates
in a single block**.

CUDA provides the `threadIdx` structure, which stores the local coordinates of
the thread in its block. The global coordinates of the latter are stored in the
`blockIdx` structure. There are also other variables that can be used to access
the lengths of the edges of both grid and blocks, as summarized in the table
below.

| Structure | Members | What it is |
| --------------- | --------------- | --------------- |
| threadIdx | threadIdx.x<br>threadIdx.y<br>threadIdx.z | x thread index in the block<br> y thread index in the block<br>z thread index in the block |
| blockIdx |  blockIdx.x<br>blockIdx.y<br>blockIdx.z |   x block index in the grid<br> y block index in the grid<br>z block index in the grid |
|   |   |  | | 
| blockDim  |  blockDim.x<br>blockDim.y<br>blockDim.z  | lenght&nbsp;in&nbsp;threads&nbsp;of&nbsp;the&nbsp;x&nbsp;edge&nbsp;of&nbsp;the&nbsp;block <br>  lenght&nbsp;in&nbsp;threads&nbsp;of&nbsp;the&nbsp;y&nbsp;edge&nbsp;of&nbsp;the&nbsp;block <br>  lenght&nbsp;in&nbsp;threads&nbsp;of&nbsp;the&nbsp;x&nbsp;edge&nbsp;of&nbsp;the&nbsp;block  |
| gridDim  |  gridDim.x<br>gridDim.y<br>gridDim.z  | lenght&nbsp;in&nbsp;blocks&nbsp;of&nbsp;the&nbsp;x&nbsp;edge&nbsp;of&nbsp;the&nbsp;grid <br>  lenght&nbsp;in&nbsp;blocks&nbsp;of&nbsp;the&nbsp;y&nbsp;edge&nbsp;of&nbsp;the&nbsp;grid <br>  lenght&nbsp;in&nbsp;blocks&nbsp;of&nbsp;the&nbsp;x&nbsp;edge&nbsp;of&nbsp;the&nbsp;grid  |



The idea now is to associate each entry (or a group of entries) of the input buffer to
a single thread, so that, when the kernel is launched, each thread will work on
its portion of the data, which will be processed in parallel at the same time.
To begin with, let us consider a one-dimensional problem, for example the sum of
two arrays. The one-dimensional nature of the task suggests to
organize the threads in similar fashion, using a one-dimensional grid of 1d
blocks. For example, 4 blocks of 4 threads each would look like this:

![1dgrid](1dgrid.png)

Now the "local" block-wise thread index has to be combined with the "global"
grid-wise block index, to index the entries of the buffer. From the figure above
it is evident that the correct expression is:
```
int idx = blockDim.x*blockIdx.x + threadIdx.x
```
In this way we have a global index for each thread in the whole grid. This index
can be used to biunivocally associate a chunk of the buffer to each thread, which
can then operate in parallel with all the others to perform its task.
The same logic also holds for 2D and 3D mapping, as you will see in the
exercises.


#### Sum of two buffers

```c
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>



__global__ void sum(const int *a, const int *b, int *c, const size_t dim) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < dim; i += stride) {
    c[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[]) {

  size_t dim = strtoll(argv[1], NULL, 10);
  size_t byte_size = dim * sizeof(int);

  int *host_a = (int *)malloc(byte_size);
  int *host_b = (int *)malloc(byte_size);
  int *host_c = (int *)malloc(byte_size);

  int *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, byte_size);
  cudaMalloc(&dev_b, byte_size);
  cudaMalloc(&dev_c, byte_size);

  for(size_t i = 0; i<dim; i++){
        host_a[i] = 4; host_b[i] = 5;
    }
  cudaMemcpy(dev_a, host_a, byte_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, host_b, byte_size, cudaMemcpyHostToDevice);

  int nthreads = 2048*2048;
  int block_size = 256;
  int grid_size =  (int)ceil(nthreads/block_size);
  sum<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, dim);

  cudaMemcpy(host_c, dev_c, byte_size, cudaMemcpyDeviceToHost);

  printf("The first 10 entries of the buffers are:\n");

  for ( int i = 0; i < 10; i++) {
    printf("a[%lu]= %d,\t b[%lu] = %d, \t, c[%lu] = %d\n", i, host_a[i], i, host_b[i], i,
           host_c[i]);
  };

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  free(host_a);
  free(host_b);
  free(host_c);

  return 0;
}

```


> [!NOTE]
> The full kernel calling syntax is:
> ```
> kernel<<<grid_size, block_size, sharedMemBytes,stream>>>
> ```
> We have just discussed the first two parameters, the third one will be briefly
> mentioned at the end of the next session and the fourth one will be discussed
> in the advanced CUDA course.


## Timing kernels

There are several ways to time code running on the GPU. One can use CPU timing
routines, making sure that the kernel actually finished running, or one can use a tool provided by NVIDIA called
`event` to facilitate kernel timing. Consider the following example:

```c
//event variables
cudaEvent_t t_in, t_f;
cudaEventCreate(&t_in);
cudaEventCreate(&t_f);

//recording events
cudaEventRecord(t_in);
kernel<<<....>>>(args);
cudaEventRecord(t_f);

//wait for final event to finish recording
cudaEventSynchronize(end);
// !NOTE the time is in milliseconds
cudaElapsedTime(&duration,start,stop);

cudaEventDestroy(start); 
cudaEventDestroy(stop); 

```
 
Finally, one can get the times from the profiler, see `profiling.md` for more
informations on this. 


## Shared memory

As discussed in the slides, there is a memory hierarchy in the GPU. The
global memory, accessible by all the threads in the grid, is the slowest. On the
other hand, each SM has a much faster (and smaller) memory bank, called "shared
memory".  
This memory can be used to load some data from the global memory before usage
and it's especially useful when this data has to be reused by multiple threads
and to avoid uncoalesced access to the GPU.  
You will play around with shared memory in one of the exercises.
As an example, consider the following kernel, that "smooths" an array, by
performing the operation:  
`a[i] -> (a[i+i]+a[i]+a[i-1])/3`
The code below implements this operation using **static** shared memory, i.e.
the amount of shared memory requested is known at compile time.
Notice that the code only works if the array can be covered by an integers
number of blocks. In real cases one has to deal with the fact that sometimes
there are threads in the last block not associated with any entry of the buffer.
You will see more of this in the many-body course.


```c
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// global variable to use statically allocated shared memory
#define THREADS_PER_BLOCK 128
#define one_third 0.333333333333
__global__ void smooth(float * array, const size_t dim)
{
    __shared__ float  shTile[THREADS_PER_BLOCK+2];

   size_t idx = blockIdx.x*blockDim.x+threadIdx.x;

   shTile[threadIdx.x+1] = array[idx];
   size_t prev = idx == 0 ? dim -1 : idx-1;
   size_t next = idx == dim-1 ? 0 : idx +1;

   if(threadIdx.x==0) shTile[threadIdx.x] = array[prev];
   if(threadIdx.x==blockDim.x-1) shTile[threadIdx.x+2] = array[next];
// IMPORTANT: you need to synchronize the threads to be sure they all finished
// loading
   __syncthreads();
   array[idx] = one_third*(shTile[threadIdx.x]+shTile[threadIdx.x+1]+shTile[threadIdx.x+2]);


}

int main(int argc, char *argv[]){

    size_t dim=1024;
    if (dim%THREADS_PER_BLOCK != 0){
        printf("This code only works if dim %% THREADS_PER_BLOCK == 0\n");
        exit(1);
    }
    float * host_a = (float*)malloc(dim*sizeof(int));
    float * dev_a;
    cudaMalloc(&dev_a,dim*sizeof(int));

    for(size_t i =0; i<dim;i++){
        host_a[i] = 3;
    }

    for(size_t i =0; i<dim;i++){
        printf("%g ", host_a[i]);
    }

    printf("\n");
    cudaMemcpy(dev_a, host_a, dim*sizeof(int), cudaMemcpyHostToDevice);
    smooth<<<dim/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dev_a,dim);


    cudaMemcpy(host_a, dev_a, dim*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i =0; i<dim;i++){
        printf("%g ", host_a[i]);
    }

    printf("\n");
    return 0;
}
```
### Dynamic shared memory

In the code above the kernel uses a static shared memory which requires the size
of the allocated memory to be known at compile time. It is also possible to use
dynamically allocated shared memory, as follows.
Inside the kernel declare as:
```c
extern __shared__ float sh[]

```
and then at kernel launch you need to specify the amount of memory that you want
to allocate:

```c
kernel<<<grid_size,block_size,sharedMemBytes>>>(args)

```
Notice that you can only allocate one buffer with dynamical shared memory. If
you need to use more buffers, one solution is to gather them in a structure:
```c
// define a structure containing the buffers you want to use
Struct ShMem
{
    float shFloat[128];
    int   shInt[32];
};

__global__ void kernel(args)
{
    // declare that structure as shared
    extern __shared__ ShMem sh[];

    // access the buffer entries
    sh->shFloat[idx];
    sh->shInt[idx];


}
```

Of course, now when you launch the kernel you need to ask for a number of bytes
equal to the memory size of the structure.

### Memory banks

Shared memory is actually divided in smaller memory chunks, called _banks_, that
can be accessed at the same time. This is done in order to increase the maximum
memory bandwidth available.
However, if multiple threads in the same warp try to access the same
bank at the same time, the access is serialized, and there is a loss in
performance. This goes under the name of _bank conflict_.
More precisely, there are 32 banks and each of them contains a range of the
addresses of the shared memory. During a memory transaction, each bank can
provide the address of a 4 bytes "word" (for example an integer or a float).
If multiple threads require variables in the same memory bank these accesses
are serialized and there is a conflict.
For example, assume to allocate `__shared__ float sh[N*N]`, the bank of each entry
`i` (`i=0,1...`) is given by:
`bank = i mod 32`
Now, during a memory transaction a warp access the shared memory, i.e. `32`
threads access the shared memory together. The optimal situation is when all
threads requests floats from different banks, so that they can be served at the
same time. Contiguous access is always ok:
```
a = sh[idx];
```
In this case each thread requests an address living in a different bank. Trouble
begins when there is a striding access:
```
a = sh[idx*32];
```
In this case the threads `0,1,...,31` in a single warp access elements
`0,32,...992` which are all in bank `0`: this is a 32-way bank conflict, and
will arise whenever the stride is an integer multiple of `32`, i.e `stride mod
32 =0`.
The typical solution is to pad the shared memory increasing the stride so that the requests do not
conflict anymore.


