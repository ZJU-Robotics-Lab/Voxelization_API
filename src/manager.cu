/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <thread>
using namespace std;

GPUTransformer::GPUTransformer (float* point_host_, int size_, int* x_, int* y_, int* height_, int max_length_, int num_x_, int num_y_, int num_height_, int enough_large_) {
  point_host = point_host_;
  h_max_length = max_length_;
  h_num_x = num_x_;
  h_num_height = num_height_;
  h_num_y = num_y_;
  enough_large = enough_large_;
  size = size_* 3 * sizeof(float);
  d_size = size_;
  int grid_size = num_x_ * num_y_ * num_height_ * enough_large * sizeof(int);
  d_grid_size = num_x_ * num_y_ * num_height_ ;
  cudaMalloc((void**) &x, d_size * sizeof(int));
  cudaMalloc((void**) &y, d_size * sizeof(int));
  cudaMalloc((void**) &height, d_size * sizeof(int));

  // auto t1 = std::chrono::high_resolution_clock::now();
  cudaMalloc((void**) &point_device, size);

  // auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "memcpy took "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
  //           << " milliseconds\n";

  cudaMemcpy(point_device, point_host, size, cudaMemcpyHostToDevice);
  cudaMemcpy(y, y_, d_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(height, height_, d_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(x, x_, d_size * sizeof(int), cudaMemcpyHostToDevice);
}

void GPUTransformer::transform() {
  dim3 blockSize(256);
  dim3 gridSize((d_size + blockSize.x - 1) / blockSize.x);
  point2gridmap<<<gridSize, blockSize>>>(point_device, x, y, height, d_size, h_max_length, h_num_x, h_num_y, h_num_height);
  cudaDeviceSynchronize();
}

void GPUTransformer::retreive(float* point_transformed) {
  int pt_count = 0;
  int index = 0;
  int x_h[d_size] = {0};
  int y_h[d_size] = {0};
  int height_h[d_size] = {0};
  int counter[d_grid_size] = {0};

  cudaMemcpy(x_h, x, d_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(y_h, y, d_size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(height_h, height, d_size * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < d_size; i++)
  {
    point_transformed[3*(y_h[i] + x_h[i] * h_num_y + height_h[i] * h_num_y * h_num_x) + 0] = point_host[i];
    point_transformed[3*(y_h[i] + x_h[i] * h_num_y + height_h[i] * h_num_y * h_num_x) + 1] = point_host[i + d_size];
    point_transformed[3*(y_h[i] + x_h[i] * h_num_y + height_h[i] * h_num_y * h_num_x) + 2] = 1;
  }

  cudaFree(point_device);
  cudaFree(height);
  cudaFree(y);
  cudaFree(x);
}

GPUTransformer::~GPUTransformer() {
  cudaFree(point_device);
  cudaFree(height);
  cudaFree(y);
  cudaFree(x);
}
