#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <torch/script.h>

inline double mse(torch::Tensor y_true, torch::Tensor y_pred){
    return torch::mean(torch::pow(2, y_true - y_pred)).item<double>();
};

inline torch::Tensor mse_prime(torch::Tensor y_true, torch::Tensor y_pred){
    return 2 * (y_pred - y_true) / y_true.size(0);
}

inline double cross_entropy(torch::Tensor y_true, torch::Tensor y_pred) {
    return -torch::sum(y_true * torch::log(y_pred)).item<double>();
}

inline torch::Tensor cross_entropy_softmax_prime(torch::Tensor y_true, torch::Tensor y_pred) {
    return y_pred - y_true;
}

#endif //LOSSES_HPP
