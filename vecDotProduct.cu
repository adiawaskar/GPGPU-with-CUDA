#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

//initialization
const  int  N = 2047;
const int blockSize = 1024;
 
const int gridSize = 4 ;

__global__ void vecMulAdd(float* a, float* b, float* c, int n, float* result){
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    //multiplication
    if(id<n){
        c[id] = a[id] * b[id];
    }
    __syncthreads();

    //addition
    __shared__ float sdata[blockSize*sizeof(float)];
    
    
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
__global__ void vecSum(float *result, float *sum, int n){
    __shared__ float sdata[gridSize*sizeof(float)]; 
    /* SHARED MEMORY SHOULD BE THE SIZE OF NUMBER OF THREADS IN A BLOCK WHICH FOR THIS KERNEL 
    WILL BE gridSize*sizeof(float) and blockDim.x variable will hold the value gridSize   */
 
 
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
 
    if(id<n){
        sdata[threadIdx.x] = result[id];
    }
    __syncthreads();
 
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1){
        if(threadIdx.x<s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
 
    if(threadIdx.x == 0){
        sum[blockIdx.x] = sdata[0];
    }
 
 
    /* The above code is literally the code from previous kernel with array names changed and the 
    element wise multiplication removed */ 
}
int main(){

    float *h_a, *h_b, *h_c, *h_sum, *h_r;
    float *d_a, *d_b, *d_c, *d_sum, *d_r;
    size_t size = N*sizeof(float);

    //memory allocation
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_sum = (float*)malloc(gridSize*sizeof(float));
    h_r = (float*)malloc(1*sizeof(float));

    cudaMalloc (&d_a,size);
    cudaMalloc (&d_b, size);
    cudaMalloc (&d_c, size);
    cudaMalloc (&d_sum, gridSize*sizeof(float));
    cudaMalloc (&d_r, 1*sizeof(float));

    int i;
    for(i=0; i<N; i++){
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }

    cudaMemcpy( d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, size, cudaMemcpyHostToDevice);

    
    //kernel call
    vecMulAdd<<<gridSize, blockSize>>>( d_a, d_b, d_c, N, d_sum); 
    vecSum<<<1, gridSize>>>(d_sum, d_r, gridSize);

    cudaMemcpy( h_sum, d_sum, gridSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy( h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaMemcpy( h_r, d_r, 1*sizeof(float), cudaMemcpyDeviceToHost);

   
    // printing each partial sum after first reduction : Just for visualization
    cout<<"Partial sums after first kernel :  "<<endl;
    for( i=0; i<gridSize; i++){
        cout<<h_sum[i]<<",";
    }
    cout<<endl;

    //print result
    cout<<"Final result after adding all partial sums : "<<h_r[0]<<endl;

    //free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}