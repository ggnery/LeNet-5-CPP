#ifndef MAX_POOLING_HPP
#define MAX_POOLING_HPP

#include "layer.hpp"
#include <ATen/core/TensorBody.h>
#include <cstdint>
#include <vector>

class MaxPooling: public Layer {
private:
    std::vector<int64_t> kernel_size;
    torch::Tensor indices;    
    int stride;
public:
    //Constructor
    MaxPooling(std::vector<int64_t> kernel_size, int stride): kernel_size(kernel_size), stride(stride) {};

    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);
};

#endif // MAX_POOLING_HPP