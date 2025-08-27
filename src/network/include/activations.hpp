#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "activation.hpp"
#include "math_utils.hpp"

class Tanh: public Activation{
    public:

    //Constructor
    Tanh(): Activation(
        [](torch::Tensor x) { return torch::tanh(x); },
        [](torch::Tensor x) { return tanh_prime(x); }
    ) {}

};

class Sigmoid: public Activation{
    public:

    // Constructor
    Sigmoid(): Activation (
        [](torch::Tensor x) { return torch::sigmoid(x); }, 
        [](torch::Tensor x) { return sigmoid_prime(x); }
    ) {}
};

#endif // ACTIVATIONS_HPP