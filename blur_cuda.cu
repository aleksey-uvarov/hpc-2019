#include <stdio.h>
#include <math.h>

// __global__ void blur(float * d_in, float * d_out)
// {
//     // int block_x = blockIdx.x;
//     // int block_y = blockIdx.y;
//     int x_glob;
//     int y_glob;
//     int x_total_dim = blockDim.x * gridDim.x;
//     int y_total_dim = blockDim.y * gridDim.y;
//     int location;

//     x_glob = blockDim.x * blockIdx.x + threadIdx.x;
//     y_glob = blockDim.y * blockIdx.y + threadIdx.y;

//     location = y_glob * x_total_dim + x_glob;

//     d_out[location] = 0.36 * d_in[location];

//     if (x_glob > 0)
//     {
//         d_out[location] += 0.12 * d_in[location - 1];
//         if (y_glob >0)
//         {
//             d_out[location] += 0.04 * d_in[location - 1 - x_total_dim];
//         }
//         if (y_glob < (y_total_dim - 1))
//         {
//             d_out[location] += 0.04 * d_in[location - 1 + x_total_dim];
//         }


//     }

//     if (x_glob < (x_total_dim - 1))
//     {
//         d_out[location] += 0.12 * d_in[location + 1];
//         if (y_glob >0)
//         {
//             d_out[location] += 0.04 * d_in[location + 1 - x_total_dim];
//         }
//         if (y_glob < (y_total_dim - 1))
//         {
//             d_out[location] += 0.04 * d_in[location + 1 + x_total_dim];
//         }

//     }

//     if (y_glob > 0)
//     {
//         d_out[location] += 0.12 * d_in[location - x_total_dim];
//     }
//     if (y_glob < (y_total_dim - 1))
//     {
//         d_out[location] += 0.12 * d_in[location + x_total_dim];   
//     }

//     // if (x_glob == 0)
//     // {
//     //     d_out[location] = 1;
//     // }

// }


int main()
{
    const int N=32;
    const int M=N;

    const int ARRAY_SIZE = N * M;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    
    // const int Niter = 10;
    float x;
    
    //size_t counter = 0;
    float h_in[ARRAY_SIZE];
    
    FILE * readfile;
    readfile = fopen("Lenna_R.txt", "r");
    for(int i=0; i<10; i++)
    {
        scanf(readfile, "%f", &x);
//         h_in[i] = x;
        printf("%f", x);
    }
    fclose(readfile);

 
    // FILE * writefile;


    // writefile = fopen("out_blur.txt", "w");

    

    // fscanf(readfile, "%f", x);
    // printf("%f\n", x);
    // float h_start[ARRAY_SIZE];

    // for(int i=0; i<N; i++)
    // {
    //     for(int j=0; j<M; j++)
    //     {
    //         fscanf(readfile,"%f", &h_start[i * M + j]);
    //         if (j< M-1)
    //         {
    //             fscanf(readfile," ");
    //         }
    //     }
    //     fscanf(readfile, "\n");
    // }


    // float h_out[ARRAY_SIZE];


    // float * d_in;
    // float * d_out;

    // cudaMalloc((void **) &d_in, ARRAY_BYTES);
    // cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // cudaMemcpy(d_in, h_start, ARRAY_BYTES, cudaMemcpyHostToDevice);
    
    // for (int i=0; i<Niter; i++)
    // {
    //     blur<<<dim3(16,16), dim3(N/16, M/16)>>>(d_in, d_out);
    //     blur<<<dim3(16,16), dim3(N/16, M/16)>>>(d_out, d_in);
    // }
    


    // cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++)
    // {
    //     for(int j=0; j<M; j++)
    //     {
    //         fprintf(writefile,"%e\t", h_out[i * M + j]);
    //     }
    //     fprintf(writefile, "\n");
    // }

    // fclose(writefile);

    // cudaFree(d_in);
    // cudaFree(d_out);

    return 0;
}

