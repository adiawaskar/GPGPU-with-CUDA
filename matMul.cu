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

    //randomly generates a matrix
    for(int i = 0; i < N*N; i++){
        m[i] = rand() % 10;
    }
}

int main(){
    // Set square matrix dimension(2^10 *2^10 default)
    int N = 1 << 1;
    size_t size = N*N*sizeof(int);

    int *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    //initialising a and b
    init_matrix(a, N);
    init_matrix(b, N);

    //allocating grid and block sizes
    int threads = 16;
    int blocks = (N + threads - 1)/threads;

    //allocating memory in kernel
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    matMul<<< BLOCKS,THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    //Matrix a
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout<<a[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    //Matrix b
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout<<b[i*N + j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    //Resulting Matrix c
    cout<<"Result: "<<endl;
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            cout<<c[i*N + j]<<" ";
        }
        cout<<endl;
    }

    return 0;
}