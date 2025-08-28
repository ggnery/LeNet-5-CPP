#include "include/softmax.hpp"
#include <ATen/ops/softmax.h>

torch::Tensor Softmax::forward(torch::Tensor input) {
    return torch::softmax(input, 0);
}

//It is assumed that softmax is used with cross-entropy
torch::Tensor Softmax::backward(torch::Tensor output_gradient, double eta){
    return output_gradient;
}  