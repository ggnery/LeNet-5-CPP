#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "sequential.hpp"
#include <functional>

class Network { 
    public:

    Sequential sequential;
    std::function<double(torch::Tensor, torch::Tensor)> loss;
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_prime;
    double eta;
    size_t epochs;

    Network(
        Sequential&& sequential,
        std::function<double(torch::Tensor, torch::Tensor)> loss,
        std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_prime,
        double eta,
        size_t epochs
    ): sequential(std::move(sequential)), loss(loss), loss_prime(loss_prime), eta(eta) {};

    void train(torch::Tensor x_train, torch::Tensor y_train, bool verbose = false);
    torch::Tensor eval(torch::Tensor);
    double accuracy(torch::Tensor x_train, torch::Tensor y_train);
    //double mean_cost(torch::Tensor x_train, torch::Tensor y_train);

};

#endif //NETWORK_HPP