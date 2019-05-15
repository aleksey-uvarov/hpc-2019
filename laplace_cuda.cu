#include <stdio.h>
#include <math.h>

__global__ void heat_step(float * d_out, float * d_in)
{
    // int block_x = blockIdx.x;
    // int block_y = blockIdx.y;
    int x_glob;
    int y_glob;
    int x_total_dim = blockDim.x * gridDim.x;
    int y_total_dim = blockDim.y * gridDim.y;
    int location;

    x_glob = blockDim.x * blockIdx.x + threadIdx.x;
    y_glob = blockDim.y * blockIdx.y + threadIdx.y;

    location = y_glob * x_total_dim + x_glob;

    d_out[location] = 0;

    if (x_glob > 0)
    {
        d_out[location] += 0.25 * d_in[location - 1];
    }

    if (x_glob < (x_total_dim - 1))
    {
        d_out[location] += 0.25 * d_in[location + 1];
    }

    if (y_glob > 0)
    {
        d_out[location] += 0.25 * d_in[location - x_total_dim];
    }
    if (y_glob < (y_total_dim - 1))
    {
        d_out[location] += 0.25 * d_in[location + x_total_dim];   
    }

    if (x_glob == 0)
    {
        d_out[location] = 1;
    }

}


int main()
{
    const int N=200;
    const int M=200;

    const int ARRAY_SIZE = N * M;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    
    const int Niter = 1000;
    
    size_t counter = 0;
    
 
    FILE * writefile;
    writefile=fopen("out_laplace.txt", "w");

    float h_start[ARRAY_SIZE];
    for(int i=0; i<ARRAY_SIZE; i++)
    {
        h_start[i] = 0;
    }
    float h_out[ARRAY_SIZE];


    float * d_in;
    float * d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_start, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    while (counter<Niter)
    {
        heat_step<<<dim3(10,10), dim3(N/10,M/10)>>>(d_in, d_out);

        heat_step<<<dim3(10,10), dim3(N/10,M/10)>>>(d_out, d_in);

        counter=counter+2;        
    }

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for(int i=0; i<N; i++)
    {
        for(int j=0; j<M; j++)
        {
            fprintf(writefile,"%e\t", h_out[i * M + j]);
        }
        fprintf(writefile, "\n");
    }

    fclose(writefile);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

