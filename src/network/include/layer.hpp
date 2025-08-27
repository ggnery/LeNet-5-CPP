#ifndef LAYER_HPP
#define LAYER_HPP

#include <torch/script.h>

class Layer{
    public:
        torch::Tensor input;
        torch::Tensor output;

        virtual torch::Tensor forward(torch::Tensor input) = 0;
        virtual torch::Tensor backward(torch::Tensor output_gradient, double eta) = 0;
};

#endif // LAYER_HPP