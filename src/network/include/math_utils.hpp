#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <torch/script.h>

inline torch::Tensor tanh_prime(torch::Tensor x){
    return 1 - torch::pow(2, torch::tanh(x));
}

inline torch::Tensor sigmoid_prime(torch::Tensor x){
    torch::Tensor s = torch::sigmoid(x);
    return s * (1 - s);
}

#endif // MATH_UTILS_HPP