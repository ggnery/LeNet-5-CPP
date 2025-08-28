#include "include/max_pooling.hpp"

torch::Tensor MaxPooling::forward(torch::Tensor input) {
    this->input = input;
    auto [result, indices] = torch::max_pool2d_with_indices(input.unsqueeze(0), this->kernel_size, this->stride);
    this->indices = indices;

    return result.squeeze(0);
}

torch::Tensor MaxPooling::backward(torch::Tensor output_gradient, double eta){
    return torch::max_unpool2d(
        output_gradient.unsqueeze(0),
        this->indices,
        {this->input.size(1), this->input.size(2)}
    ).squeeze(0);
}

