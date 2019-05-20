/*
 * This program uses the host CURAND API. 
 * I copied most of it from cuRAND 
 * documentation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <math.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
    

__global__ void float_to_int(int * devInts, float * devData)
{
    int x_glob;
    int y_glob;
    int x_total_dim = blockDim.x * gridDim.x;
    //int y_total_dim = blockDim.y * gridDim.y;
    int location;

    x_glob = blockDim.x * blockIdx.x + threadIdx.x;
    y_glob = blockDim.y * blockIdx.y + threadIdx.y;

    location = y_glob * x_total_dim + x_glob;
    
    devInts[location] = (int) floor(devData[location] * 10);
}

__global__ void simple_histo(int * devInts, int * devHistogram)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = devInts[myId];
    int myBin = myItem;
    
    atomicAdd(&(devHistogram[myBin]), 1);
}

int main(int argc, char *argv[])
{
    int nx = 1024;
    int ny = 1024;
    //int n_threads = 10;
    int n = nx * ny;
    int i;
    curandGenerator_t gen;
    float *devData;
    //float *hostData;
    int *histogram;
    int *devInts;
    //int *hostInts;
    int *devHistogram;
    

    /* Allocate n floats on host */
//     hostData = (float *)calloc(n, sizeof(float));
    //hostInts = (int *)calloc(n, sizeof(int));
    histogram = (int *)calloc(10, sizeof(int));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));
    
    CUDA_CALL(cudaMalloc((void **)&devInts, n*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&devHistogram, 10*sizeof(int)));

    

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
    
    /* Turn them into random integers */
    float_to_int<<<dim3(32,32), dim3(nx/32,ny/32)>>>(devInts, devData);

    /* Make a simple histogram */
    simple_histo<<<dim3(1024), dim3(n / 1024)>>>(devInts, devHistogram);
    

    /* Copy device memory to host */
//     CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
//         cudaMemcpyDeviceToHost));

    

    
//     CUDA_CALL(cudaMemcpy(hostInts, devInts, n * sizeof(int),
//     cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(histogram, devHistogram, 10 * sizeof(int),
    cudaMemcpyDeviceToHost));

    /* Show result */
//     for(i = 0; i < n; i++) {
//         printf("%1.4f ", hostData[i]);
//     }
//     printf("\n");
//     
//     for(i = 0; i < n; i++) {
//         printf("%d ", hostInts[i]);
//     }
//     printf("\n");
    
    for(i = 0; i<10; i++) 
    {
        printf("%d ", histogram[i]);
    }
    printf("\n");
    

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    CUDA_CALL(cudaFree(devInts));
    CUDA_CALL(cudaFree(devHistogram));
    free(histogram);
    //free(hostData);    
    //free(hostInts);    
    return EXIT_SUCCESS;
}
