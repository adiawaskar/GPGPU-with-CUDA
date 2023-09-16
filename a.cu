#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
using namespace std;

const int N = 2;

__global__ void matAdd3d(int *a, int *b, int *c, int N ){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;

    if(row < N && col < N && layer < N){
        c[layer*N*N + row*N + col] = a[layer*N*N + row*N + col] + b[layer*N*N + row*N + col];
    }
}

void init_matrix(int *m, int N){
    for(int i = 0; i < N*N*N; i++){
        m[i] = 1;
    }
}

int main(){
    int size = N*N*N*sizeof(int);

    int *h_a = new int[N*N*N];
    int *h_b = new int[N*N*N];
    int *h_c = new int[N*N*N];

    init_matrix(h_a, N);
    init_matrix(h_b, N);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //allocating grid and block sizes
    int threads = 16;
    int blocks = (N + threads - 1)/threads;

    //allocating memory in kernel
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    matAdd3d<<< BLOCKS, THREADS>>> (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);


    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_a[i*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    } 
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_b[i*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_c[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}