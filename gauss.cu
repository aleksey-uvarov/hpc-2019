#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
     *       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


__global__ void generate_uniform_kernel(curandState *state,
                                        int n_points_per_thread, 
                                        int *result)
{
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    float x;
    float y;
    float z;
    float r2;
    
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n_points_per_thread; i++) {
        x = curand_uniform(&localState) * 4 - 2;
        y = curand_uniform(&localState) * 4 - 2;
        r2 = pow(x, 2) + pow(y, 2);
        z = curand_uniform(&localState);
        if(z < exp(-1 * r2)) {
            count++;
        }
//         if (z > 0.5)
//         {
//             count++;
//         }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void shmem_reduce( int *d_out,  int *d_in)
{
    extern __shared__ int sdata[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid]=d_in[myId];
    int s = blockDim.x / 2;
    
    while(s>0)
    {
        if (tid<s)
        {
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
        s=( int)s/2;
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] =sdata[0];
    }
}


int main()
{
    int n_threads = 1024;
    int n_points_per_thread = 1000000;
    curandState *devStates;
//     int total;
    int *devResults;
    int *devIntermediate;
    int *devReduced;
    int *hostResults;
    int *hostReduced;    
    
    hostResults = ( int *)calloc(n_threads, sizeof(  int));
    hostReduced = ( int *)calloc(n_threads, sizeof(  int));
    
    
    CUDA_CALL(cudaMalloc((void **)&devResults, n_threads * sizeof( int)));    
    CUDA_CALL(cudaMalloc((void **)&devReduced, n_threads * sizeof( int)));
    CUDA_CALL(cudaMalloc((void **)&devIntermediate, n_threads * sizeof( int)));        
    
    CUDA_CALL(cudaMalloc((void **)&devStates, n_threads * sizeof(curandState)));
    
    CUDA_CALL(cudaMemset(devResults, 0, n_threads * sizeof( int)));
    CUDA_CALL(cudaMemset(devReduced, 0, n_threads * sizeof( int)));
    
    setup_kernel<<<1, n_threads>>>(devStates);
    
    generate_uniform_kernel<<<1, n_threads>>>(devStates, n_points_per_thread, devResults);
    
    shmem_reduce<<<n_threads / 32, 32, n_threads * sizeof(int)>>>(devIntermediate,devResults);
    shmem_reduce<<<32, 32, 32 * sizeof(int)>>>(devReduced,devIntermediate);
    
    
    CUDA_CALL(cudaMemcpy(hostResults, devResults, n_threads*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaMemcpy(hostReduced, devReduced, n_threads*sizeof(int), cudaMemcpyDeviceToHost));
    
    
//     for (int i=0; i<n_threads; i++)
//     {
//         printf("%d ", hostResults[i]);
//     }
//     printf("\n");
    
/*    
    for (int i=0; i<n_threads; i++)
    {
        printf("%d ", hostReduced[i]);
    }
    printf("\n");*/
    
    printf("Total area: %1.7f \n", (float) hostReduced[0] / (float) n_points_per_thread / (float) n_threads * 16);
    
    CUDA_CALL(cudaFree(devResults));
    CUDA_CALL(cudaFree(devReduced));
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devIntermediate));
    free(hostReduced);
    free(hostResults);
    
    
    return EXIT_SUCCESS;  
    
}
