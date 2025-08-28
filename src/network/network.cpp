#include "include/network.hpp"
#include <ATen/ops/argmax.h>
#include <ATen/ops/equal.h>
#include <iostream>

torch::Tensor Network::eval(torch::Tensor input){
    return this->sequential.forward(input);
}

void Network::train(torch::Tensor x_train, torch::Tensor y_train, bool verbose){
    int train_size = x_train.size(0);
    for(int epoch = 0; epoch < this->epochs; epoch++){
        double loss_sum = 0;
        for (int i = 0; i < train_size; i++){
            torch::Tensor x = x_train[i];
            torch::Tensor y = y_train[i];

            torch::Tensor y_pred = this->eval(x);
            loss_sum += this->loss(y, y_pred);

            //backward
            torch::Tensor output_gradient = this->loss_prime(y, y_pred);
            this->sequential.backward(output_gradient, eta);
        }

        if (verbose){
            std::cout << "Loss in epoch " << epoch + 1 <<": " << loss_sum/train_size << std::endl;
        }
    }
}

double Network::accuracy(torch::Tensor x_test, torch::Tensor y_test) {
    long sum = 0;
    for (int i=0; i<y_test.size(0); i++) {
        torch::Tensor x = x_test[i];
        torch::Tensor y = y_test[i];

        if (torch::equal(torch::argmax(eval(x)), torch::argmax(y))) sum++;
    }
    return static_cast<double>(sum) / y_test.size(0);
}