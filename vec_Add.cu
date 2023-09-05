#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void vecAdd(float* a, float* b, float* c, int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if(id<n){
        c[id] = a[id] * b[id];
    }
}
int main(){
    int N = 4;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t size = N*sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    cudaMalloc (&d_a,size);
    cudaMalloc (&d_b, size);
    cudaMalloc (&d_c, size);

    int i;
    for(i=0; i<N; i++){
        h_a[i] = 1.0;
        h_b[i] = 2.0;
    }

    cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)N/blockSize);

    vecAdd<<<gridSize, blockSize>>>( d_a, d_b, d_c, N);

    cudaMemcpy( h_c, d_c, size, cudaMemcpyDeviceToHost);

    int sum=0;
    for( i=0; i<N; i++){
        sum += h_c[i];
    }

    for( i=0; i<N; i++){
        cout<<h_c[i]<<" ";
    }
    cout<<endl;

    cout<<"Final result: "<<sum<<endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}