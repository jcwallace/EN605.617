//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <assert.h>
    
// CUDA Libraries
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
    
#define MAX 50   
#define index(i,j,ld) (((j)*(ld))+(i))

__global__
void init(unsigned int seed, curandState_t* states) {
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;  
    curand_init(seed, thread_idx, 0, &states[thread_idx]);
}
 
__global__
void random(curandState_t* states, float* numbers) {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}
    
void printMatrix(float*P,int n)
{
    if( n > 10 ){
        printf("\nOmitting (too large)...\n");
        return;
    }
    
    int i,j;
    for(i=0;i<n;i++){
        printf("\n");
        for(j=0;j<n;j++)
            printf("%f ",P[index(i,j,n)]);
    }
    printf("\n");
}

void printMatrix(int n, double*P)
{
  //printf("\n %f",P[1]);
  int i,j;
  for(i=0;i<n;i++){

      printf("\n");

      for(j=0;j<n;j++)
          printf("%f ",P[index(i,j,n)]);
  }
  printf("\n");
}
    
void main_MM(int mDim){
    int matrixDim = mDim;
    int mSize = mDim*mDim;
    // ---------------------------------- //
    // ----  Random Matrix Generate  ---- //
    // ---------------------------------- //
    float *a_host = (float*)malloc(mSize*sizeof(float));
    float *b_host = (float*)malloc(mSize*sizeof(float));
    float *c_host = (float*)malloc(mSize*sizeof(float));
    float *A, *B, *C;
    curandState_t* states1;
    curandState_t* states2;
        
    cudaMalloc((void**) &states1, mSize * sizeof(curandState_t));
    init<<<matrixDim, matrixDim>>>(42, states1);
    cudaMalloc((void**) &states2, mSize * sizeof(curandState_t));
    init<<<matrixDim, matrixDim>>>(time(0), states2);
    
    cudaMalloc((void**) &A, sizeof(float)*mSize );
    cudaMalloc((void**) &B, sizeof(float)*mSize );
    cudaMalloc((void**) &C, sizeof(float)*mSize );
    
    random<<<matrixDim, matrixDim>>>(states1, A);
    random<<<matrixDim, matrixDim>>>(states2, B);
    
    cudaMemcpy(a_host, A, mSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_host, B, mSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("___A___");
    printMatrix(a_host,matrixDim);
    
    printf("___B___");
    printMatrix(b_host,matrixDim);
    
    // ----------------------- //
    // ----  Matrix Mult  ---- //
    // ----------------------- //
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t ret;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    ret = cublasSgemm
                (
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    matrixDim,
                    matrixDim,
                    matrixDim,
                    &alpha,
                    A,
                    matrixDim,
                    B,
                    matrixDim,
                    &beta,
                    C,
                    matrixDim
                );
    
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
    }
    auto stop = std::chrono::high_resolution_clock::now(); 
    float delta = std::chrono::duration<double,std::milli>(stop - start).count();
    printf("-- Matrix Multiplication (%dx%d) [%f ms]\n", mDim, mDim, delta);
    
    cudaMemcpy(c_host, C, mSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("___A*B___ ");
    printMatrix(c_host,matrixDim);
    
    free( a_host );  free( b_host );  free ( c_host );
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaDeviceReset();
}
    
void main_eigenSolve()
{
    // ------------------------ //
    // ----  Eigen Solver  ---- //
    // ------------------------ //
    printf("\nSolving for Eiganvalues");
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    const int m = 3;
    const int lda = m;

    
    double A[lda*m] = { 3.5, 0.5, 0, 0.5, 3.5, 0, 0, 0, 2.0};
    double lambda[m] = { 2.0, 3.0, 4.0};
    double V[lda*m]; // eigenvectors
    double W[m]; // eigenvalues
    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;
    int info_gpu = 0;

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize
                        (
                            cusolverH,
                            jobz,
                            uplo,
                            m,
                            d_A,
                            lda,
                            d_W,
                            &lwork
                        );
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    // compute spectrum
    cusolver_status = cusolverDnDsyevd
                        (
                            cusolverH,
                            jobz,
                            uplo,
                            m,
                            d_A,
                            lda,
                            d_W,
                            d_work,
                            lwork,
                            devInfo
                       );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    
    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    
    printf("eigenvalue(s) = \n");
    for(int i = 0 ; i < m ; i++){
        printf("W[%d] = %E\n", i+1, W[i]);
    }
                         
       
    // check eigenvalues
    double lambda_sup = 0;
    for(int i = 0 ; i < m ; i++){
        double error = fabs( lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error)? lambda_sup : error;
    }
    printf("|lambda - W| = %E\n", lambda_sup);
    
    // free resources
    if (d_A ) cudaFree(d_A);
    if (d_W ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaDeviceReset();
}
    
    
int main(int argc, char** argv)
{
	// read command line arguments
	int matrixDim = 3;
    	
	if (argc >= 2) {
		matrixDim = atoi(argv[1]);
	}
	// validate command line arguments
	if (matrixDim == 0) {
		matrixDim = 3;
		
		printf("Warning: Size specified too small (using default 3)\n");
	}
    
    printf("Matrix Dim: %d\n",matrixDim);
    
    main_MM(matrixDim);
    main_MM(matrixDim*10);
    main_MM(matrixDim*100);
    
    main_eigenSolve();
    
}
