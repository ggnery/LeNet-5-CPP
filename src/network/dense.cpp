#include "include/dense.hpp"

// Constructor
Dense::Dense(long input_size, long output_size){
    this->weights = torch::randn({output_size, input_size}) / std::sqrt(static_cast<float>(input_size)); // W ~ N(0, 1/sqtr(n_in))
    this->bias = torch::randn({output_size, 1}); // b ~ N(0, 1)
}

//Forward
torch::Tensor Dense::forward(torch::Tensor input) {
    this->input = input;
    return torch::matmul(this->weights, input) + this->bias; // W*x + b
}

// Backward
torch::Tensor Dense::backward(torch::Tensor output_gradient, double eta) {
    torch::Tensor weight_gradient = torch::matmul(output_gradient, torch::transpose(this->input, 0 ,1)); // ∂E/∂W = ∂E/∂Y * X^T
    torch::Tensor bias_gradient = output_gradient; // ∂E/∂B = ∂E/∂Y 
    torch::Tensor input_gradient =  torch::matmul(torch::transpose(this->weights, 0, 1), output_gradient); // ∂E/∂X = W^T * ∂E/∂Y 

    // SGD
    this->weights -= eta * weight_gradient;
    this-> bias -= eta * bias_gradient;

    return input_gradient;
}


