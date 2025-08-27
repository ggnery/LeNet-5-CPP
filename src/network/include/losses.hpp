#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <torch/script.h>

template <typename T>
inline T mse(torch::Tensor y_true, torch::Tensor y_pred){
    return torch::mean(torch::pow(2, y_true - y_pred));
};

inline torch::Tensor mse_prime(torch::Tensor y_true, torch::Tensor y_pred){
    return 2 * (y_pred - y_true) / y_true.size(0);
}

template <typename T>
inline T cross_entropy(torch::Tensor y_true, torch::Tensor y_pred) {
    return torch::sum((-y_true * torch::log(y_pred)) - ((1 - y_true) * torch::log(1 -y_pred)));
}

inline torch::Tensor cross_entropy_prime(torch::Tensor y_true, torch::Tensor y_pred) {
    return ((1 - y_true)/(1 - y_pred) - (y_true/y_pred) ) / y_true.size(0);
}

#endif //LOSSES_HPP
