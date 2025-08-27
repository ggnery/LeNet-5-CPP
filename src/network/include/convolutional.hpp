#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "layer.hpp"

class Convolutional: public Layer {
    public:
    torch::IntArrayRef input_shape; // (CxHxW)
    torch::IntArrayRef output_shape; // (dxH'xW')
    torch::IntArrayRef kernels_shape; // (dxCxkxk)

    int input_channels;
    int n_kernels;

    torch::Tensor kernels;
    torch::Tensor bias;

    Convolutional(torch::IntArrayRef input_shape, int kernel_size, int channels);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);

};

#endif // CONVOLUTION_HPP