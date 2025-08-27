#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/layer.hpp"
#include "network/include/losses.hpp"
#include "network/include/network.hpp"
#include "network/include/convolutional.hpp"
#include "dataset/include/mnist.hpp"
#include <ATen/ops/one_hot.h>
#include <iostream>

int main(){
    torch::Tensor images = read_mnist_images("./data/t10k-images.idx3-ubyte");
    std::cout << images.sizes() << std::endl;
    torch::Tensor labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte");
    std::cout << labels.sizes() << std::endl;
    
    torch::Tensor x = torch::reshape(images, {images.size(0), 1, 28, 28});
    torch::Tensor y = torch::one_hot(labels, 10);
    y = torch::reshape(y, {y.size(0), 10, 1});
    
    // Convolutional conv = Convolutional({3,5,5}, 2, 3);
    // //Create a Sequential class here
    // std::vector<std::unique_ptr<Layer>> layers;
    // layers.push_back(std::make_unique<Dense>(2, 3));
    // layers.push_back(std::make_unique<Sigmoid>());
    // layers.push_back(std::make_unique<Dense>(3, 1));
    // layers.push_back(std::make_unique<Sigmoid>());

    // Network network = Network(std::move(layers), mse, mse_prime, 0.1, 10000); 
    // network.train(x, y, true);
    
    // for (int i = 0; i < x_test.size(0); i++){
    //     std::cout << network.eval(x_test[i]) << std::endl;
    // }
    
    return 0;
}
