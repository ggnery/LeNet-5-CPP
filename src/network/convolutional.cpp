#include "include/convolutional.hpp"
#include <ATen/ops/randn.h>

Convolutional::Convolutional(std::tuple<int, int, int> input_shape, int kernel_size, int n_kernels){
    auto [channels, input_height, input_width] = input_shape; // Input is 3D: (CxHxW)
    this->n_kernels = n_kernels; // n_kernels = d
    this->input_shape = input_shape;
    this->input_channels = channels;

    // Y = I - K +1
    // The channel of the output is the same as the number of kernels
    this->output_shape = {n_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1}; // Output is 3D: (dxH'xW')
    
    this->kernels_shape = {n_kernels, channels, kernel_size, kernel_size}; // Kernel is 4D: (dxCXkxk)

    this->kernels = torch::randn({
        std::get<0>(this->kernels_shape), 
        std::get<1>(this->kernels_shape), 
        std::get<2>(this->kernels_shape), 
        std::get<3>(this->kernels_shape)
    });

    // bias is 3D: (dxH'xW')
    this->bias = torch::randn({
        std::get<0>(this->output_shape), 
        std::get<1>(this->output_shape), 
        std::get<2>(this->output_shape)
    });
}