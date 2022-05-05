// Common Includes
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>


// Constants
#define NUM_RUNS 4

// Squared matrices are tested within a certain range (e.g. 1024x1024, 2048x2048, 4096x4096)
#define MINSIZE (1024)
#define MAXSIZE (4*1024)

// Set the alpha and beta values for the cuBLAS and clBlas libraries. Note that the myGEMM kernels
// for simplicity only support alpha values of 1 and beta values of 0.
#define ALPHA 1.0f
#define BETA 0.0f

#define GPU_NAME "Tesla K80"
#define GPU_CLOCK 0.823 // Core clock in GHz
#define GPU_CORES 4992 // Total number of CUDA cores
//#define GPU_MOD 2 // Fused multiply-add

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

void libcublas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);

void libclblas(float* A, float* B, float* C,
               int K, int M, int N,
               int timerID);

void mycublas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID);

void myclblas(float* A, float* B, float* C,
              int K, int M, int N,
              int timerID);

class CUDAMatMult {

    public:
        int* a,b,c;
        CUDAMatMult();
        ~CUDAMatMult();
    
};

class NvidiaMatMult {

    public:
        int* a,b,c;

    NvidiaMatMult();
    ~NvidiaMatMult();
};

