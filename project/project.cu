//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
    
#define NUM_ELEMENTS 100000    

__global__
void add(int * arr1, int * arr2, int * out)
{    
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = arr1[thread_idx] + arr2[thread_idx];
}
    

__global__  
void sub(int * arr1, int * arr2, int * out){
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = arr1[thread_idx] - arr2[thread_idx];
}
    

__global__
void mult(int * arr1, int * arr2, int * out){
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = arr1[thread_idx] * arr2[thread_idx];
}
    

__global__ 
void mod(int * arr1, int * arr2, int * out){
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[thread_idx] = arr1[thread_idx] % arr2[thread_idx];
}

       
void main_math(int blockSize, int numBlocks) {
    
    // Main Assignment Function calls
    int array1[NUM_ELEMENTS];
    int array2[NUM_ELEMENTS];
    int output[NUM_ELEMENTS];
    
    for( int i = 0; i<NUM_ELEMENTS; i++){
        array1[i] = i;
        array2[i] = rand() % 4;
    }
    
    int *arr1,*arr2,*out;
    
    int arraySize_bytes = (sizeof(unsigned int) * NUM_ELEMENTS);                             
    cudaMalloc((void **)&arr1, arraySize_bytes);
    cudaMalloc((void **)&arr2, arraySize_bytes);
    cudaMalloc((void **)&out, arraySize_bytes);
                   
    auto start = std::chrono::high_resolution_clock::now();                                
    cudaMemcpy(arr1,array1,arraySize_bytes,cudaMemcpyHostToDevice);                                
    cudaMemcpy(arr2,array2,arraySize_bytes,cudaMemcpyHostToDevice);                                 
                                 
    add<<<numBlocks,blockSize>>>(arr1,arr2,out);
    sub<<<numBlocks,blockSize>>>(arr1,arr2,out);
    mult<<<numBlocks,blockSize>>>(arr1,arr2,out);
    mod<<<numBlocks,blockSize>>>(arr1,arr2,out);
    
    cudaMemcpy(output,out,arraySize_bytes,cudaMemcpyDeviceToHost);
    
    auto stop = std::chrono::high_resolution_clock::now(); 
    float delta = std::chrono::duration<double,std::milli>(stop - start).count();
    printf("-- Global memory (Baseline) [%f ms]\n", delta);
    
    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(out);
    
}

 
void main_math_stream(int blockSize, int numBlocks){
    
    int arraySize_bytes = (sizeof(unsigned int) * NUM_ELEMENTS);
    int *host_arr1, *host_arr2, *host_out;
    int *dev_arr1, *dev_arr2, *dev_out;
                                    
    cudaStream_t stream; 
    cudaStreamCreate(&stream);
                                                                    
    cudaMalloc((void **)&dev_arr1, arraySize_bytes);
    cudaMalloc((void **)&dev_arr2, arraySize_bytes);
    cudaMalloc((void **)&dev_out, arraySize_bytes);
    
    cudaHostAlloc((void **)&host_arr1, arraySize_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_arr2, arraySize_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_out, arraySize_bytes, cudaHostAllocDefault);
    
    for( int i = 0; i<NUM_ELEMENTS; i++){
        host_arr1[i] = i;
        host_arr2[i] = rand() % 4;
    }
        
    auto start = std::chrono::high_resolution_clock::now();                                
                                    
    cudaMemcpyAsync(dev_arr1, host_arr1, arraySize_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_arr2, host_arr2, arraySize_bytes, cudaMemcpyHostToDevice, stream);                                
                                    
    add<<<NUM_ELEMENTS,numBlocks,blockSize,stream>>>(dev_arr1,dev_arr2,dev_out);
    sub<<<NUM_ELEMENTS,numBlocks,blockSize,stream>>>(dev_arr1,dev_arr2,dev_out);
    mult<<<NUM_ELEMENTS,numBlocks,blockSize,stream>>>(dev_arr1,dev_arr2,dev_out);
    mod<<<NUM_ELEMENTS,numBlocks,blockSize,stream>>>(dev_arr1,dev_arr2,dev_out);
    
    cudaMemcpyAsync(host_out, dev_out, arraySize_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    auto stop = std::chrono::high_resolution_clock::now(); 
    float delta = std::chrono::duration<double,std::milli>(stop - start).count();
    printf("-- Stream and Event [%f ms]\n", delta);
    
    cudaFreeHost(host_arr1);
    cudaFreeHost(host_arr2);
    cudaFreeHost(host_out);
    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_out);
    cudaDeviceReset();
}
    
    
int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = 256;
	int blockSize = 256;
    int numBlocks = totalThreads/blockSize;
    	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    
    numBlocks = totalThreads/blockSize;
    
    printf("Thread Count: %d\n",totalThreads);
    printf("Block Size: %d\n",blockSize);
    printf("Array Size: %d\n",NUM_ELEMENTS);
    
    main_math(blockSize, numBlocks);
    main_math_stream(blockSize, numBlocks);
    
}
