#include "project.h"       

void generateMatrix(float* mat, int l, int w){
/*
*   Fill Matrices with rand values (matrix is treated as a 1d array)
*/
    for (int i=0; i<l*w; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char** argv)
{
    double peak = GPU_CLOCK * GPU_CORES * GPU_MOD;

    printf("##    cuBLAS on '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak);

    for (int size=MINSIZE; size<=MAXSIZE; size=size*2) {
        // Set the performance counters to zero
        for (int t=0; t<NUM_TIMERS; t++) {
            timers[t].t = 0.0;
            timers[t].kflops = 0;
        }

        // Set the matrices to be squared (change this to get rectangular matrices)
        const int k = size;
        const int m = size;
        const int n = size;

        float* A = (float*)malloc(m*k*sizeof(float*));
        float* B = (float*)malloc(k*n*sizeof(float*));
        float* C = (float*)malloc(m*n*sizeof(float*));
        float* goldC = (float*)malloc(MAXSIZE*MAXSIZE*sizeof(float*));
        generateMatrix(A,m,k);
        generateMatrix(B,k,n);


    }

    return 0;
}