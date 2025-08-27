#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "layer.hpp"

class Reshape: Layer {
    public:
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    Reshape(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);

};

#endif // RESHAPE_HPP