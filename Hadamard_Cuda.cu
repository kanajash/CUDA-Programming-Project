#include <cstddef>
#include <cstdio>
#include <stdlib.h>
#include <stdio.h>


void dump_arr(size_t numCol, size_t numRow, float* Arr)
{
    for (size_t j = 0; j < numCol; ++j) {
      for (size_t i = 0; i < numRow; ++i) {
      printf("%.2f ", Arr[i * numCol + j]);
    }
    printf("\n");
  }
}

__global__
void cuda_hadamard(size_t numCol, size_t numRow, float* Z, float* X, float* Y)
{
  size_t threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadColId = blockIdx.y * blockDim.y + threadIdx.y;    

  Z[threadRowID * numCol + threadColId] = X[threadRowID * numCol + threadColId] * Y[threadRowID * numCol + threadColId];
}



int main()
{
  const size_t ARRAY_DIM = 4096;
  const size_t ARRAY_BYTES = ARRAY_DIM * ARRAY_DIM * sizeof(float);
  size_t NUM_EXEC = 30;

  // Array Initialization
  float* X;
  float* Y;
  float* Z;

  // Array Malloc
  cudaMallocManaged(&X, ARRAY_BYTES);
  cudaMallocManaged(&Y, ARRAY_BYTES);
  cudaMallocManaged(&Z, ARRAY_BYTES);

  // get gpu ID
  int device = -1;
  cudaGetDevice(&device);

  // Mem advise
  cudaMemAdvise(X, ARRAY_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(X, ARRAY_BYTES, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
  cudaMemAdvise(Y, ARRAY_BYTES, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(Y, ARRAY_BYTES, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);

  // "prefetch data" to create CPU page memory
  cudaMemPrefetchAsync(X, ARRAY_BYTES, cudaCpuDeviceId, NULL);
  // "prefetch data" to create CPU page memory
  cudaMemPrefetchAsync(Y, ARRAY_BYTES, cudaCpuDeviceId, NULL);
  // "prefetch data" to create GPU page memory
  cudaMemPrefetchAsync(Z, ARRAY_BYTES, device, NULL);

  // initialize array contents
  for (size_t i = 0; i < ARRAY_DIM; ++i) {
    for (size_t j = 0; j < ARRAY_DIM; ++j) {
      X[ARRAY_DIM * i + j] = 1;
      Y[ARRAY_DIM * i + j] = 1;
    }
  }

  // "Prefetch data" from CPU-GPU
  cudaMemPrefetchAsync(X, ARRAY_BYTES, device, NULL);
  cudaMemPrefetchAsync(Y, ARRAY_BYTES, device, NULL);

  // setup CUDA kernel
  // https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/2D-grids.html
  size_t threadDimBlock = 32;

  dim3 blockShape = dim3(threadDimBlock, threadDimBlock);
  // Algorithm for dim3 gridShape gathered from:
  // https://selkie.macalester.edu/csinparallel/modules/GPUProgramming/build/html/CUDA2D/CUDA2D.html
  // https://medium.com/@harsh20111997/cuda-programming-2d-convolution-8476300f566e
  dim3 gridShape = dim3( (ARRAY_DIM + threadDimBlock - 1) / threadDimBlock, (ARRAY_DIM + threadDimBlock - 1)/threadDimBlock );

  for (size_t i = 0; i < NUM_EXEC; ++i) {
    cuda_hadamard <<< gridShape, blockShape >>> (ARRAY_DIM, ARRAY_DIM, Z, X, Y);
  }

  cudaDeviceSynchronize();

  // "Prefetch data" from GPU-CPU
  cudaMemPrefetchAsync(X, ARRAY_BYTES, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(Y, ARRAY_BYTES, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(Z, ARRAY_BYTES, cudaCpuDeviceId, NULL);


  // dump_arr(ARRAY_DIM, ARRAY_DIM, X);
  // printf("\n\n");
  // dump_arr(ARRAY_DIM, ARRAY_DIM, Z);

  // error checking
  size_t errCount = 0;

  for (size_t i = 0; i < ARRAY_DIM; ++i) {
    for (size_t j = 0; j <ARRAY_DIM; ++j ) {
      if (X[i * ARRAY_DIM + j] * Y[i * ARRAY_DIM + j] != Z[i * ARRAY_DIM + j]) {
        errCount++;
      }
    }
  }

  printf("Array Dimension: %lux%lu\n", ARRAY_DIM, ARRAY_DIM);
  printf("Thread Block Dimension: %lux%lu\n", threadDimBlock, threadDimBlock);
  printf("Total error count: %lu", errCount);

  // Free memory
  cudaFree(X);
  cudaFree(Y);
  cudaFree(Z);


  return 0;
}
