#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
using namespace std;

const int N = 2;

__global__ void matMul3d(int *a, int *b, int *c, int N){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;
    if(col<N && row < N && layer < N){
        int temp = 0;
        for(int i = 0; i < N; i++){
            temp += a[layer*N*N + row*N + i]*b[layer*N*N + i*N + col];
        }

        c[layer*N*N + row*N + col] = temp;
    }

}

void init_matrix(int *m, int N){
    for(int i = 0; i < N*N*N; i++){
        m[i] = rand() % 10;
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

    dim3 THREADS(N, N, N);
    dim3 BLOCKS(N, N, N);

    matMul3d<<< BLOCKS, THREADS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cout<<"Matrix A: "<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_a[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    } 
    cout<<"Matrix B: "<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_b[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<"Result:"<<endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j<N; j++){
            for(int k = 0; k<N; k++){
                cout<<h_c[i*N*N + j*N + k]<<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }

    return 0;
}