#include <bits/stdc++.h>
#include "conversion.h"
#include "edges.h"
using namespace std;

using matrix = vector<vector<double>>;

__global__ matrix edgeDetect(matrix R, matrix G, matrix B, result)
{
    col = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y * blockDim.y + threadIdx.y;

    matrix kernel_X = {{-1, 0, 1},
                       {-2, 0, 2},
                       {-1, 0, 1}};
    matrix kernel_Y = {{-1, -2, -1},
                       {0, 0, 0},
                       {1, 2, 1}};

    matrix gray_image = cvtGray(R, G, B);
    matrix res = conv2D(gray_image, kernel_X, 8);
    matrix res2 = conv2D(gray_image, kernel_Y, 8); 
    matrix final_image(gray_image.size(), vector<double>(gray_image[0].size(), 0));
    
    if(row < res.size() && col < res[0].size()){
        final_image[row][col] = sqrt(pow(res[row][col], 2) + pow(res2[row][col], 2));
            if (final_image[row][col] > 0.05)
                final_image[row][col] = 1;
            else
                final_image[row][col] = 0;
        
        result[row][col] = final_imaage[row][col];
    }
    __syncthreads();
    //return final_image;
    
}
