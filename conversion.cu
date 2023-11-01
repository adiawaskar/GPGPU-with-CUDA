#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

using matrix = vector<vector<double>>;

__device__ matrix cvtGrey( matrix R, matrix G, matrix B){
    matrix res(R.size(), vector<double>(R[0].size(), 0));
    if( row < R.size() && col < R[0].size()){
        res[row][col] = float(R[row][col] + G[row][col] + B[row][col]) / 3;
    }
    return res;
}

__device__ matrix conv2d(matrix image, matrix kernel){
    start_c = col - MASK_OFFSET;
    start_r = row - MASK_OFFSET;
    matrix ret(height, vector<double>(width, 0));

    if(row < height && col < width){

        for(int i = 0; i < kernel.size(); i++){

            for(int j = 0; j < kernel[0].size(); j++){

                if((start_r + i)>=0 && (start_r + i) < image.size()){

                    if((start_c + j) >=0 && (start_c + j) < image[0].size()){

                        ret[row][col] += image[start_r + i][start_c + j] * kernel[i][j];

                    }
                }
            }
        }
    }

    return ret;
}

__device__ Mat create2d(image){

    Mat final_image(image.size(), image.at(0).size(), CV_64FC1);
    if(row < final_image.rows && col < final_image.col){
        final_img.at<double>(row, col) = image.at(row).at(col);
    }

    return final_image;
}