#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
// #include <stdexcept>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


__global__ void prepare_function(float * d_out, int n_points,
                                 float x_min, float x_max)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float x = x_min + (x_max - x_min) * id / n_points;
    d_out[id] = exp(-pow(x, 2));
//     d_out[id] = (float) id;
}


// __global__ void blelloch_reduce(float * d_in, int n_points)
// {
//     /* Assuming n_points is a power of two */
//     int n_current = 2;
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     
//     while(n_current <= n_points)
//     {
//         if ((id + 1) % n_current == 0)
//         {
//             d_in[id] += d_in[id - n_current/2];
//         }
//         __syncthreads();
//         n_current = n_current * 2;
//     }
//     
// }


__global__ void blelloch_reduce_step(float * d_in, int n_points, int stride_step)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if ((id + 1) % stride_step == 0)
    {
        d_in[id] += d_in[id - stride_step/2];
    }
}


__global__ void kill_last(float * d_in, int n_points)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;       
   
    if (id == n_points - 1)
    {
        d_in[id] = 0;
    }
}


/*
__global__ void blelloch_downsweep(float * d_in, int n_points)
{
    int n_current = n_points;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float tmp;
    
    if (id == n_points - 1)
    {
        d_in[id] = 0;
    }
    __syncthreads();
    
    while(n_current >= 2)
    {
        if ((id + 1) % n_current == 0)
        {
            tmp = d_in[id];
            d_in[id] += d_in[id - n_current/2];
            d_in[id - n_current/2] = tmp;
        }
        n_current = n_current / 2;
        __syncthreads();
    }
}*/



__global__ void blelloch_downsweep_step(float * d_in, int n_points, int stride_step)
{
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;       
    float tmp;
  
 
    if ((id + 1) % stride_step == 0)
    {
        tmp = d_in[id];
        d_in[id] += d_in[id - stride_step/2];
        d_in[id - stride_step/2] = tmp;
    }
//     __syncthreads();
}


int main(int argc, char* argv[])
{
    float minus_infty = -8;
    float x_max = 0;
    int n_blocks = 1024;
    int n_points_per_block = 1024;
    int n_points = n_points_per_block * n_blocks;
    int stride_step;
    
    float dx;
    float *devFunVals;
    float *hostFunVals;
    float *hostFunVals2;
    
    if (argc > 1)
    {
        sscanf(argv[1], "%f", &x_max);
//         printf("%f\n", x_max);
        if (x_max < minus_infty)
        {
            printf("0\n");
            return 0;
        }
    }
    else
    {
        printf("Usage: ./scan <number> \n");
        return 0;
    }
    dx = (x_max - minus_infty) / (float) n_points;
//     printf("dx: %e\n", dx);
//     printf("n_points: %d\n", n_points);
//     
    if (n_points < 0 || ((n_points & (n_points - 1)) != 0))
    {
        printf("n_points is not a power of two\n");
        return 1;
    }
    
    hostFunVals = (float *)calloc(n_points, sizeof(float));
    hostFunVals2 = (float *)calloc(n_points, sizeof(float));
    
    CUDA_CALL(cudaMalloc((void **)&devFunVals, n_points*sizeof(float)));

    
    prepare_function<<<n_blocks, n_points_per_block>>>(devFunVals, n_points, minus_infty, x_max);
    
//     blelloch_reduce<<<n_blocks, n_points_per_block>>>(devFunVals, n_points);

    stride_step = 2;
    while(stride_step <= n_points)
    {
        blelloch_reduce_step<<<n_blocks, n_points_per_block>>>(devFunVals, n_points, stride_step);        
        
        stride_step = stride_step * 2;
    }
    stride_step = n_points;

    
    CUDA_CALL(cudaMemcpy(hostFunVals, devFunVals, n_points*sizeof(float), cudaMemcpyDeviceToHost));
    
    kill_last<<<n_blocks, n_points_per_block>>>(devFunVals, n_points);
    
    while(stride_step >= 2)
    {
        blelloch_downsweep_step<<<n_blocks, n_points_per_block>>>(devFunVals, n_points, stride_step);
        stride_step = stride_step / 2; 
    }


    CUDA_CALL(cudaMemcpy(hostFunVals2, devFunVals, n_points*sizeof(float), cudaMemcpyDeviceToHost));

//     for(int i=0; i<n_points; i++)
//     {
//         printf("%1.4f \n", hostFunVals[i]);
//     }
//     printf("=========\n");
//     
//     for(int i=0; i<n_points; i++)
//     {
//         printf("%1.4f \n", hostFunVals2[i]);
//     }
//     printf("\n");
//     printf("Func value: %1.5f\n", hostFunVals[n_points - 1]);
    printf("Integral value: %1.5e\n", hostFunVals2[n_points - 1] * dx);
    
    
    
    
    
    
    
    
    
    
    
    
    free(hostFunVals);
    free(hostFunVals2);

    CUDA_CALL(cudaFree(devFunVals));


    
    
    return 0;    
}
