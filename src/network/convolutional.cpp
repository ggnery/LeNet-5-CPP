#include "include/convolutional.hpp"
#include <ATen/Functions.h>


Convolutional::Convolutional(torch::IntArrayRef input_shape, int kernel_size, int n_kernels){
    // Input is 3D: (CxHxW)
    int channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];

    this->n_kernels = n_kernels; // n_kernels = d
    this->input_shape = input_shape;
    this->input_channels = channels;

    // Y = I - K +1
    // The channel of the output is the same as the number of kernels
    this->output_shape = {n_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1}; // Output is 3D: (dxH'xW')
    this->kernels_shape = {n_kernels, channels, kernel_size, kernel_size}; // Kernel is 4D: (dxCxkxk)

    this->kernels = torch::randn(this->kernels_shape);
    this->bias = torch::randn(this->output_shape); // bias is 3D: (dxH'xW')
}

torch::Tensor Convolutional::forward(torch::Tensor input) {
    this->input = input;
    this->output = this->bias.clone();

    //            n
    //    Yi = Bi+∑ Xj ⋆ Kj, i = 0..d
    //            j
    for(int i=0; i < this->n_kernels; i++){ // i = 0..d where d is the number of kernels   
        for(int j=0; j < this->input_channels; j++){ // #  = 0..n where n is channel size of the input

            //Apply cross-correlation but with conv2d function 
            this->output[i] += torch::conv2d(
                this->input[j].unsqueeze(0), 
                torch::flip(this->kernels[i][j], {0, 1}).unsqueeze(0),
                {},
                1,
                "valid"
            ).squeeze(0);
        }
    }

    return this->output;
}
