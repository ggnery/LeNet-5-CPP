#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "layer.hpp"
#include <ATen/core/TensorBody.h>
#include <vector>
#include <functional>
#include <memory>

class Network { 
    public:

    std::vector<std::unique_ptr<Layer>> layers;
    std::function<double(torch::Tensor, torch::Tensor)> loss;
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_prime;
    double eta;
    size_t epochs;

    Network(std::vector<std::unique_ptr<Layer>> layers,
        std::function<double(torch::Tensor, torch::Tensor)> loss,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_prime,
        double eta,
        size_t epochs);

    void train(torch::Tensor x_train, torch::Tensor y_train, bool verbose = false);
    torch::Tensor eval(torch::Tensor);
    double accuracy(torch::Tensor x_train, torch::Tensor y_train);
    //double mean_cost(torch::Tensor x_train, torch::Tensor y_train);

};

#endif //NETWORK_HPP