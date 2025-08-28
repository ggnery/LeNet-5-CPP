#include "include/sequential.hpp"
#include <iostream>

torch::Tensor Sequential::forward(torch::Tensor input) {
    std::cout << this->device << std::endl;
    for (const auto& layer : this->layers){
        input = layer->forward(input);
    }
    return input;
}

torch::Tensor Sequential::backward(torch::Tensor output_gradient, double eta) {
    for(int j = this->layers.size() - 1; j >= 0; j--) {
        output_gradient = this->layers[j]->backward(output_gradient, eta);
    }
    return output_gradient;
}

