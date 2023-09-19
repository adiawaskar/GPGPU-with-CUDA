#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

#define MASK_DIM 2
#define MASK_OFFSET (MASK_DIM/2)


__global__ void conv_3d(int *matrix, int *mask, int * result, int N){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;

    int start_c = col - MASK_OFFSET;
    int start_r = row - MASK_OFFSET;
    int start_l = layer - MASK_OFFSET;

    if(col < N && row < N && layer < N){
        int temp = 0;
        for(int i = 0; i < MASK_DIM; i++){

            for(int j = 0; j < MASK_DIM; j++){

                for(int k = 0; k < MASK_DIM; k++){

                    if(((start_l + i) >=0) && ((start_l + i) < N)){

                        if(((start_r + j) >= 0) &&  ((start_r + j) < N)){

                            if(((start_c + k) >= 0) && ((start_c + k) < N)){

                                temp += matrix[(start_l + i)*N*N + (start_r + j)*N + start_c + k] * mask[(i * MASK_DIM * MASK_DIM) + (j * MASK_DIM) + k];
                            }
                        }
                    }
                }
            }
        }
        result[layer*N*N + row*N + col] = temp;
    }
}

void init_matrix(int *m, int N){
    for(int i = 0; i < N*N*N; i++){
        m[i] = 1;
    }
}

int main(){
    int N = 2;

    //initialise and allocate matrix and result memory
    int size_n = N*N*N*sizeof(int);

    int *matrix = new int[N*N*N];
    int *result = new int[N*N*N];
    init_matrix(matrix, N);

    //initialise and allocate mask memory
    int size_m = MASK_DIM * MASK_DIM * MASK_DIM * sizeof(int);

    int *mask = new int[MASK_DIM * MASK_DIM * MASK_DIM];
    init_matrix(mask, MASK_DIM);

    int *d_matrix, *d_mask, *d_result;
    cudaMalloc(&d_matrix, size_n);
    cudaMalloc(&d_mask, size_m);
    cudaMalloc(&d_result, size_n);

    cudaMemcpy(d_matrix, matrix, size_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_m, cudaMemcpyHostToDevice);

    dim3 block_dim(N, N, N);
    dim3 grid_dim(N, N, N);

    conv_3d<<< grid_dim, block_dim>>>(d_matrix, d_mask, d_result, N );
    cudaMemcpy(result, d_result, size_n, cudaMemcpyDeviceToHost);

    cout<<"Input Matrix : "<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<matrix[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    } 
    cout<<"Mask: "<<endl;
    for(int i = 0; i < MASK_DIM; i++){
        for(int j = 0; j<MASK_DIM; j++){
            for(int k = 0; k<MASK_DIM; k++){
                cout<<mask[i*MASK_DIM*MASK_DIM + j*MASK_DIM + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<"Result:"<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<result[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}