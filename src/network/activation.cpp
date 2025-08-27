#include "include/activation.hpp"

//Constructor
Activation::Activation(std::function<torch::Tensor(torch::Tensor)> f, std::function<torch::Tensor(torch::Tensor)> f_prime) {
    this->f = f;
    this->f_prime = f_prime;
}


torch::Tensor Activation::forward(torch::Tensor input) {
    this->input = input;
    return this->f(input); // Y = f(X) 
}

torch::Tensor Activation::backward(torch::Tensor output_gradient, double eta) {
    return output_gradient * this->f_prime(this->input); // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
}