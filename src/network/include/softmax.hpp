#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "layer.hpp"

class Softmax: public Layer {
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor backward(torch::Tensor output_gradient, double eta);  
};

#endif // SOFTMAX_HPP