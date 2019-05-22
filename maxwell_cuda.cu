#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h> 

float boundary(float t) {
  float fun = 0;
  if (t < M_PI * 20)
    {
      fun = sin(t);
    }
  return fun;
}


__global__ void maxwell_step(float * d_out, float * d_in, float boundary)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id==0)
    {
        d_out[id] = boundary;
    } else 
    {
        d_out[id] = d_in[id - 1];
    }  
}


int main()
{
    int n_x = 1024;
    
    float L = 2 * M_PI * 2;
    float T = 2 * M_PI;
    
    float dx = L / n_x;
    float dt = dx;
    
    int n_t = (int) floor(T / dt);
    
    int j = 0;
    
    int n_blocks = 32;
    
    float * d_in;
    float * d_out;
    float * h_out;

    h_out = (float *)calloc(n_x, sizeof(float));

    cudaMalloc((void **)&d_in, n_x*sizeof(float));
    cudaMalloc((void **)&d_out, n_x*sizeof(float));
    
    cudaMemset(d_in, 0, n_x * sizeof(float));
    
    while (j < n_t)
    {   
        maxwell_step<<<n_blocks, n_x/n_blocks>>>(d_out, d_in, boundary(j * dt));
        j++;
        maxwell_step<<<n_blocks, n_x/n_blocks>>>(d_in, d_out, boundary(j * dt));
        j++;
    }
    cudaMemcpy(h_out, d_in, n_x*sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i=0; i<n_x; i++)
    {
        printf("%1.4f ", h_out[i]);
    }
    printf("\n");
    
    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_out);

    return 0;
}
