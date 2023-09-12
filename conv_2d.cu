#include <iostream>
#include <stdlib.h>

using namespace std;

#define MASK_DIM 3

#define MASK_OFFSET (MASK_DIM/2)

__constant__ int mask[3*3];

__global__ void conv_2d(int *matrix, int *result, int N){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    int temp = 0;

    for(int i = 0; i<MASK_DIM; i++){
        for (int j = 0; j<MASK_DIM; j++){
            if(((start_r + i)>=0)&&((start_r + i)<N)){
                if(((start_c + j)>=0)&&((start_c + j)<N)){
                    temp += matrix[(start_r + i)*N + (start_c + j)] * mask[i*MASK_DIM + j];
                }
            }
        }
    }

    result[row*N + col] = temp;
}

void init_matrix(int *m, int N){

    //randomly generates a matrix
    for(int i = 0; i < N*N; i++){
        m[i] = 1;
    }
}

int main(){
    int N = 3;

    //initialise input and output matrix and allocate memory
    int size_n = N*N*sizeof(int);

    int *matrix = new int [size_n];
    int *result = new int [size_n];
    init_matrix(matrix, N);

    //initialise mask and allocate memory
    int size_m = MASK_DIM * MASK_DIM * sizeof(int);

    int *h_mask = new int [MASK_DIM*MASK_DIM];
    init_matrix(h_mask, MASK_DIM);
    
    //allocate device vectors and their memory
    int *d_matrix, *d_result;
    cudaMalloc(&d_matrix, size_n);
    cudaMalloc(&d_result, size_n);
    //cudaMalloc(&d_mask, size_m);

    cudaMemcpy(d_matrix, matrix, size_n, cudaMemcpyHostToDevice);
    cudaMemcpy(mask, h_mask, size_m, cudaMemcpyHostToDevice);

    int threads = 16;
    int blocks = (N + threads - 1)/ threads;

    dim3 block_dim(threads, threads);
    dim3 grid_dim(blocks, blocks);

    conv_2d<<< grid_dim, block_dim>>>(d_matrix, d_result, N );
    cudaMemcpy(result, d_result, size_n, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            cout<<matrix[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            cout<<h_mask[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            cout<<result[i*N + j]<<" ";
        }
        cout<<endl;
    }

    return 0;

}