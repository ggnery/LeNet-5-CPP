#ifndef LAYER_HPP
#define LAYER_HPP

#include <torch/script.h>

class Layer{
protected:
    torch::Tensor input;
    torch::Tensor output;
    torch::Device device;
public:
    Layer(): device(
        #ifdef TORCH_CUDA_AVAILABLE
            torch::Device("cuda")
        #else
            torch::Device("cpu")
        #endif
    ) {};
    virtual ~Layer() = default;
    virtual torch::Tensor forward(torch::Tensor input) = 0;
    virtual torch::Tensor backward(torch::Tensor output_gradient, double eta) = 0;
};

#endif // LAYER_HPP