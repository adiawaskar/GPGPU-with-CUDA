#include <bits/stdc++.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "conversion.h"
using namespace std;
using matrix = vector<vector<double>>;

int main(){
    string filename = argv[2]; // Getting the filename from the command line
    string readCommand = "python3.8 ../src/read_image.py " + filename;
    int ret = system(readCommand.c_str()); // Calling the python program from the command line
    Mat im = imread(argv[2]);              // Reading the image
    int height = im.rows, width = im.cols; // Getting the height and width of the input image
    // PixelData.txt ==> Contains the pixel values of CFA
    fstream inputFile("PixelData.txt", ios::in);
    double buffer;
    int i = 0;
    vector<double> temp;
    // Making an empty image filled with zeros
    matrix image(height + 4, vector<double>(width + 4, 0));
    matrix result(height + 4, vector<double>(width + 4, 0));
    while (inputFile)
    {
        inputFile >> buffer;
        i++;
        temp.push_back((buffer));
    }
    int k = 0;
    for (int i = 2; i < height + 2; i++)
    {
        for (int j = 2; j < width + 2; j++)
        {
            image[i][j] = temp[k];
            k++;
        }
    }

    size = (height + 4, vector<double>(width + 4, 0));
    matrix d_image, d_result;
    cudaMalloc(&d_image, size_n );
    cudaMalloc(&d_result, size_n);

    cudaMemcpy(d_image, image , size_n, cudaMemcpyHostToDevice);

    int threads = 16;

    // Calculate the number of blocks needed in each dimension
    int blocks_x = (width + threads - 1) / threads; // Adjust for the width of the image
    int blocks_y = (height + threads - 1) / threads; // Adjust for the height of the image

    // Define the block dimensions
    dim3 block_dim(threads, threads, 1);

    // Create the grid dimensions
    dim3 grid_dim(blocks_x, blocks_y, 1);

    edgeDetect<<<grid_dim, block_dim>>>(R, G, B, d_result);

    cudaMemcpy(result, d_result, size_n, cudaMemcpyDeviceToHost);

    Mat edges = create2d(result);
    edges.convertTo(edges, CV_8UC3, 255.0);
    imwrite("Edges.png", edges);
    resize(edges, edges, Size(500, 500));
    imshow("Edges", edges);
    cout << "\n\nImage saved as \"Edges.png\"" << endl;
}