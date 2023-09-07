#include <iostream>
#include <cstdlib>

using namespace std;

__global__ void matMul(int *a, int *b, int *c, int N){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(col < N && row < N){
        
        int temp = 0;
        for(int i = 0; i < N; i++){
            temp += a[row*N + i]*b[i*N + col];
        }

        c[row*N + col] = temp;
    }
}

void init_matrix(int *m, int N){
    for(int i = 0; i < N*N; i++){
        m[i] = rand() % 100;
    }
}

int main(){
    // Set square matrix dimension(2^10 *2^10 default)
    int N = 1 << 10;
    size_t size = N*N*sizeof(int);

    int *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    init_matrix(a, N);
    init_matrix(b, N);

    int threads = 16;
    int blocks = (N + threads - 1)/threads;

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    matMul<<< BLOCKS,THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    cout<<c<<endl;

    return 0;
}