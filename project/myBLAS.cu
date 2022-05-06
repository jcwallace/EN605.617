#include <project.h>

// Replace the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global 
#define __placeholder__ __global__
#define __local __shared__
#define restrict __restrict__

// Replace OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()

// Replace the OpenCL get_xxx_ID with CUDA equivalents
__device__ int get_local_id(int x) {
    return (x == 0) ? threadIdx.x : threadIdx.y;
}
__device__ int get_group_id(int x) {
    return (x == 0) ? blockIdx.x : blockIdx.y;
}
__device__ int get_global_id(int x) {
    return (x == 0) ? blockIdx.x*blockDim.x + threadIdx.x : blockIdx.y*blockDim.y + threadIdx.y;
}

#include "kernels.cl"

void myblas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID) {

    // Prepare CUDA memory objects
    float* bufA = 0;
    float* bufB = 0;
    float* bufB_TR = 0; // This is the transposed version of B
    float* bufC = 0;
    cudaMalloc((void**)&bufA,    M*K*sizeof(*A));
    cudaMalloc((void**)&bufB,    K*N*sizeof(*B));
    // cudaMalloc((void**)&bufB_TR, N*K*sizeof(*B));
    cudaMalloc((void**)&bufC,    M*N*sizeof(*C));

    // Copy matrices to the GPU (memset C to erase the results of the previous run)
    cudaMemcpy((void*)bufA, (void*)A, M*K*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)bufB, (void*)B, K*N*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemset((void*)bufC, 0.0, M*N*sizeof(*C));

    // Configure the local memory (banks of 8 bytes, 48KB local memory)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Configure the thread/threadblock dimensions of the myGEMM kernel
    dim3 blocks(M/TS, N/TS);
    dim3 threads(TS, TS);

    // Start the timed loop
    double startTime = timer();
    for (int r=0; r<NUM_RUNS; r++) {

        // Run the myGEMM kernel
        matMul<<<blocks, threads>>>(M, N, K, bufA, bufB, bufC);

        // Wait for calculations to be finished
        cudaDeviceSynchronize();
    }

    // End the timed loop
    timers[timerID].t += (timer() - startTime) / (double)NUM_RUNS;
    timers[timerID].kflops += ((long)K * (long)M * (long)N * 2) / 1000;

    // Copy the output matrix C back to the CPU memory
    cudaMemcpy((void*)C, (void*)bufC, M*N*sizeof(*C), cudaMemcpyDeviceToHost);

    // Free the GPU memory objects
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(bufB_TR);
    cudaFree(bufC);
}