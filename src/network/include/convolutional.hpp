#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "layer.hpp"
#include <tuple>

class Convolutional: public Layer {
    public:
    std::tuple<int, int, int> input_shape;
    std::tuple<int, int, int> output_shape;
    std::tuple<int, int, int, int> kernels_shape;

    int input_channels;
    int n_kernels;

    torch::Tensor kernels;
    torch::Tensor bias;

    Convolutional(std::tuple<int, int, int> input_shape, int kernel_size, int channels);

};

#endif // CONVOLUTION_HPP