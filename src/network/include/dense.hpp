#ifndef DENSE_HPP
#define DENSE_HPP

#include "layer.hpp"

class Dense: public Layer{
    public:
        torch::Tensor weights;
        torch::Tensor bias;

        // Constructor
        Dense(int input_size, int output_size);
    
        torch::Tensor forward(torch::Tensor input);
        torch::Tensor backward(torch::Tensor output_gradient, double eta);
};

#endif // DENSE_HPP