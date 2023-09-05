#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void vecAdd(float* a, float* b, float* c, int n, float* result){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    //multiplication
    if(id<n){
        c[id] = a[id] * b[id];
    }
    __syncthreads();

    //addition
    __shared__ float sdata[1024*sizeof(float)];
    
    
    if(id<n){
        sdata[threadIdx.x] = c[id];
    }
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1){
        if(threadIdx.x<s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        result[blockIdx.x] = sdata[0];
    }
}
int main(){
    //initialization
    int N = 1025;
    float *h_a, *h_b, *h_c, *h_sum;
    float *d_a, *d_b, *d_c, *d_sum;
    size_t size = N*sizeof(float);

    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (int)ceil((float)N/blockSize);

    //memory allocation
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_sum = (float*)malloc(size);

    cudaMalloc (&d_a,size);
    cudaMalloc (&d_b, size);
    cudaMalloc (&d_c, size);
    cudaMalloc (&d_sum, size);

    int i;
    for(i=0; i<N; i++){
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, size, cudaMemcpyHostToDevice);

    
    //kernel call
    vecAdd<<<gridSize, blockSize>>>( d_a, d_b, d_c, N, d_sum);
    //vecAdd<<< 1, blockSize>>>( d_a, d_b, d_c, N, d_sum);

    cudaMemcpy( h_c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy( h_sum, d_sum, size, cudaMemcpyDeviceToHost);

   
    //printing each intermediate vector
    for( i=0; i<N; i++){
        cout<<h_c[i]<<" ";
    }
    cout<<endl;

    //print result
    cout<<"Final result: "<<h_sum[0]<<endl;

    //free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}