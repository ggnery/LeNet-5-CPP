#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/losses.hpp"
#include "network/include/network.hpp"
#include "network/include/convolutional.hpp"
#include "dataset/include/mnist.hpp"
#include "network/include/reshape.hpp"
#include "network/include/sequential.hpp"
#include "network/include/sequential.hpp"
#include "network/include/max_pooling.hpp"
#include <iostream>
#include <vector>

int main(){
    torch::Tensor images = read_mnist_images("./data/t10k-images.idx3-ubyte");
    torch::Tensor labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte");
    
    auto [x_train, x_test,  y_train, y_test] = preprocess_data(images, images, labels, labels);
    std::cout << "Train images sizes: "<< x_train.sizes() << std::endl;
    std::cout << "Train labels sizes: "<< y_train.sizes() << std::endl;
    std::cout << "Test images sizes: "<< x_test.sizes() << std::endl;
    std::cout << "Test labels sizes: "<< y_test.sizes() << std::endl;

    // Sequential sequential = Sequential::builder()
    //     .add<Convolutional>(std::vector<int64_t>{1, 28, 28}, 3, 5)
    //     .add<Sigmoid>()
    //     .add<Reshape>(std::vector<int64_t>{5, 26, 26}, std::vector<int64_t>{5*26*26, 1})
    //     .add<Dense>(5*26*26, 100)
    //     .add<Sigmoid>()
    //     .add<Dense>(100, 10)
    //     .add<Sigmoid>().build();
    

    // Network network = Network(std::move(sequential), cross_entropy, cross_entropy_prime, 0.1, 2); 
    // network.train(x_train, y_train, true);
    
    // double accuracy = network.accuracy(x_test, y_test);
    // std::cout << "Final model accuracy: " << accuracy << std::endl;
    return 0;
}
