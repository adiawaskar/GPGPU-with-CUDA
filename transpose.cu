#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void transpose(int *in, int *out, int n){
    int cols = blockIdx.x*blockDim.x + threadIdx.x;
    int rows = blockIdx.y*blockDim.y + threadIdx.y;

    if(rows<n && cols<n){
        for(int i=0; i<n; i++){
            out[cols*n + rows] = in[rows*n + cols];
        }
    }
}

void init_matrix(int *m, int N){

    //randomly generates a matrix
    for(int i = 0; i < N*N; i++){
        m[i] = rand() % 10;
    }
}

int main(){
    int N = 1 << 2;
    size_t size = N*N*sizeof(int);

    int *a, *b;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);

    //cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    //initialising a
    init_matrix(a, N);

    //allocating grid and block sizes
    int threads = 16;
    int blocks = (N + threads - 1)/threads;

    //allocating memory in kernel
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);


    transpose<<< BLOCKS, THREADS>>>(a, b, N);
    cudaDeviceSynchronize();

    //cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    //Matrix a
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout<<a[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    cout<<"Resulting Matrix: "<<endl;
    cout<<"Result: "<<endl;
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout<<b[i*N + j]<<" ";
        }
        cout<<endl;
    }

    cudaFree(a);
    cudaFree(b);
    return 0;
    
}