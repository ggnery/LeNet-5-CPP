#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "layer.hpp"
#include <functional>

class Activation: public Layer {
protected:
    //Calback function that computes the activation function f(X): R^n->R^n 
    std::function<torch::Tensor(torch::Tensor)> f; 
    
    //Calback function that computes the derivative of activation function f'(X): R^n->R^n 
    std::function<torch::Tensor(torch::Tensor)> f_prime; 

public:
    // Constructor
    Activation(std::function<torch::Tensor(torch::Tensor)> f, std::function<torch::Tensor(torch::Tensor)> f_prime);

    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);
};

#endif // ACTIVATION_HPP