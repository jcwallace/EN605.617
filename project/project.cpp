#include "project.h"       

profiler timers[NUM_TIMERS];

void generateMatrix(float* mat, int l, int w){
/*
*   Fill Matrices with rand values (matrix is treated as a 1d array)
*/
    for (int i=0; i<l*w; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}

double timer(void){
    struct timeval t;
    struct timezone tz;
    gettimeofday(&t,&tz);
    double etime = (double)t.tv_sec + 1.0e-6*((double)t.tv_usec);
    return etime;
}

double wtime(profiler timer){
    return timer.t;
}

double gflops(profiler timer){
    return ((double)timer.kflops/(1000.0*1000.0))/(timer.t);
}

char* readKernel(const char* filename, long* _size){
    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s\n", filename);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    fread(source, 1, size*sizeof(char), file);
    source[size] = '\0';
    fclose(file);

    // Save the size and return the source string
    *_size = (size+1);
    return source;
}

int main(int argc, char** argv)
{
    double peak = GPU_CLOCK * GPU_CORES * GPU_MOD;

    printf("## GPU '%s', peak: %.1lf GFLOPS\n", GPU_NAME, peak);

    for (int size=MINSIZE; size<=MAXSIZE; size=size*2) {
        // Set the performance counters to zero
        for (int itr=0; itr<NUM_TIMERS; itr++) {
            timers[itr].t = 0.0;
            timers[itr].kflops = 0;
        }

        // Set the matrices to be squared (change this to get rectangular matrices)
        const int k = size;
        const int m = size;
        const int n = size;

         printf("## --- %dx%dx%d ---\n", k, m, n);

        float* A = (float*)malloc(m*k*sizeof(float*));
        float* B = (float*)malloc(k*n*sizeof(float*));
        float* C = (float*)malloc(m*n*sizeof(float*));
        float* goldC = (float*)malloc(MAXSIZE*MAXSIZE*sizeof(float*));
        generateMatrix(A,m,k);
        generateMatrix(B,k,n);

        // Benchmark to compare results to
        cublas(A,B,goldC,k,m,n, NUM_TIMERS-1);
                                        
        for(int c=0; c<=2; c++){
            char name[100];
            switch(c){
                case 0: sprintf(name,"cuBLAS"); cublas(A,B,C,k,m,n,c); break;
                case 1: sprintf(name,"myBLAS_cu"); myblasCUDA(A,B,C,k,m,n,c); break;
                case 2: sprintf(name,"myBLAS_cl"); myblasCL(A,B,C,k,m,n,c); break;
            }
            
            // Print the results to screen
            double seconds = wtime(timers[c]);
            double performance = gflops(timers[c]);
            double fraction = 100.0 * performance / peak;
            printf("## [%9s] %6.3lf s --> %6.1lf GFLOPS (%2.0lf%%)\n",
                    name, seconds, performance, fraction);
        }

        free(A);
        free(B);
        free(C);
        free(goldC);
    }

    return 0;
}