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
