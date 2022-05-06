// Common Includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>


// Constants
#define NUM_RUNS 4
#define TS 32

// Squared matrices are tested within a certain range (e.g. 1024x1024, 2048x2048, 4096x4096)
#define MINSIZE (256)
#define MAXSIZE (4*1024)

// Set the alpha and beta values for the cuBLAS and clBlas libraries. Note that the myGEMM kernels
// for simplicity only support alpha values of 1 and beta values of 0.
#define ALPHA 1.0f
#define BETA 0.0f

#define GPU_NAME "Tesla K80"
#define GPU_CLOCK 0.823 // Core clock in GHz
#define GPU_CORES 4992 // Total number of CUDA cores
#define GPU_MOD 2 // Fused multiply-add

typedef struct{
	double t;
	int long long kflops;
} profiler;

// Number of timers
#define NUM_TIMERS 10

extern profiler timers[NUM_TIMERS];

double timer(void);
double wtime(profiler timer);
double gflops(profiler timer);

void cublas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);

void myblasCL(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);

void myblasCUDA(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID);

// Print device properties
// void printDevProp(cudaDeviceProp devProp)
// {
//     printf("Major revision number:         %d\n",  devProp.major);
//     printf("Minor revision number:         %d\n",  devProp.minor);
//     printf("Name:                          %s\n",  devProp.name);
//     printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
//     printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
//     printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
//     printf("Warp size:                     %d\n",  devProp.warpSize);
//     printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
//     printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
//     for (int i = 0; i < 3; ++i)
//     printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
//     for (int i = 0; i < 3; ++i)
//     printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
//     printf("Clock rate:                    %d\n",  devProp.clockRate);
//     printf("Total constant memory:         %u\n",  devProp.totalConstMem);
//     printf("Texture alignment:             %u\n",  devProp.textureAlignment);
//     printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
//     printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
//     printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//     return;
// }