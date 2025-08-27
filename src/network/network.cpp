#include "include/network.hpp"
#include <iostream>

Network::Network(
    std::vector<std::unique_ptr<Layer>> layers,
    std::function<double(torch::Tensor, torch::Tensor)> loss,
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> loss_prime,
    double eta,
    size_t epochs){
        this->layers = std::move(layers);
        this->loss = loss;
        this->loss_prime = loss_prime;
        this->eta = eta;
        this->epochs = epochs;
    }

torch::Tensor Network::eval(torch::Tensor input){
    for (const auto& layer : this->layers){
        input = layer->forward(input);
    }
    return input;
}

void Network::train(torch::Tensor x_train, torch::Tensor y_train, bool verbose){
    int train_size = x_train.size(0);
    for(size_t epoch = 0; epoch < this->epochs; epoch++){
        double loss_sum = 0;
        for (size_t i = 0; i < train_size; i++){
            torch::Tensor x = x_train[i];
            torch::Tensor y = y_train[i];

            torch::Tensor y_pred = this->eval(x);
            loss_sum += this->loss(y, y_pred);

            //backward
            torch::Tensor output_gradient = this->loss_prime(y, y_pred);
            for(size_t j = static_cast<int>(this->layers.size()) - 1; j >= 0; j--) {
                output_gradient = this->layers[j]->backward(output_gradient, this->eta);
            }
        }

        if (verbose){
            std::cout << "Loss in epoch " << epoch + 1 <<": " << loss_sum/train_size;
        }
    }
}