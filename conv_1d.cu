#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

const int N = 2059;
const int M = 2;

//setting blocksize and gridsize
int threads = 256;
int blocks = (N + threads - 1)/threads;

__global__ void conv_1d(int *array, int *mask, int *result, int N, int M){
    // calculating global thread id
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    //calculate radius of mask
    int r = M / 2;

    int start = tid - r;

    int temp = 0;

    for(int i = 0; i<M; i++){
        if(((start + i)>=0) && ((start + i)<N)){

            temp += array[start + i]*mask[i];
        }
    }
    result[tid]= temp;

}
int main(){

    //size of input array
    int size_n = N*sizeof(int);

    //size of mask
    int size_m = M*sizeof(int);

    int *h_array = new int[N];

    //initializing all elements of input array as 1
    for(int i=0; i<N; i++){
        h_array[i] = 1;
    }

    int *h_mask = new int[M];

    //initializing all elements of mask as 1
    for(int i = 0; i<M; i++){
        h_mask[i] = 1;
    }

    int *h_result = new int[N];

    //initializing device vectors
    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, size_n);
    cudaMalloc(&d_mask, size_m);
    cudaMalloc(&d_result, size_n);

    

    //copy data from cpu to gpu
    cudaMemcpy(d_array, h_array, size_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_m, cudaMemcpyHostToDevice);

    //kernel call
    conv_1d<<< threads, blocks>>>( d_array , d_mask , d_result ,  N , M);

    //copying result back from gpu to cpu
    cudaMemcpy(h_result, d_result, size_n, cudaMemcpyDeviceToHost);

    for(int i = 0; i<N; i++){
        
        cout<<h_array[i]<<" ";

    }
    cout<<endl;

    for(int i = 0; i<M; i++){
        
        cout<<h_mask[i]<<" ";

    }
    cout<<endl;

    for(int i = 0; i<N; i++){
        
        cout<<h_result[i]<<" ";

    }

    cudaFree(d_array);
    cudaFree(d_mask);
    cudaFree(d_result);
    free(h_array);
    free(h_mask);
    free(h_result);

    return 0;

}     