#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/layer.hpp"
#include "network/include/losses.hpp"
#include "network/include/network.hpp"
#include "network/include/convolutional.hpp"

int main(){
    torch::Tensor x = torch::reshape(torch::tensor({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}}), {4, 2, 1});
    torch::Tensor y = torch::reshape(torch::tensor({{0.0}, {1.0}, {1.0}, {0.0}}), {4, 1, 1});
    torch::Tensor x_test = torch::reshape(torch::tensor({{0.0, 0.0}, {0.01, 0.99}, {0.99, 0.01}, {0.85, 0.75}}), {4, 2, 1});

    //Create a Sequential class here
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Dense>(2, 3));
    layers.push_back(std::make_unique<Sigmoid>());
    layers.push_back(std::make_unique<Dense>(3, 1));
    layers.push_back(std::make_unique<Sigmoid>());

    Network network = Network(std::move(layers), mse, mse_prime, 0.1, 10000); 
    network.train(x, y, true);
    
    for (int i = 0; i < x_test.size(0); i++){
        std::cout << network.eval(x_test[i]) << std::endl;
    }
    
    return 0;
}
