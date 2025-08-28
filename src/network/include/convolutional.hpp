#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "layer.hpp"

class Convolutional: public Layer {
private:
    std::vector<int64_t> input_shape; // (CxHxW)
    std::vector<int64_t> output_shape; // (dxH'xW')
    std::vector<int64_t> kernels_shape; // (dxCxkxk)

    int input_channels;
    int n_kernels;

    torch::Tensor kernels;
    torch::Tensor bias;
public:
    Convolutional(std::vector<int64_t> input_shape, int kernel_size, int n_kernels);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);

};

#endif // CONVOLUTION_HPP