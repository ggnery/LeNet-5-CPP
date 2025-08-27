#include "include/reshape.hpp"

Reshape::Reshape(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape){
    this->input_shape = input_shape;
    this->output_shape = output_shape;
}

torch::Tensor Reshape::forward(torch::Tensor input){
    return torch::reshape(input, this->output_shape);
}

torch::Tensor Reshape::backward(torch::Tensor output_gradient, double eta){
    return torch::reshape(output_gradient, this->input_shape);
}