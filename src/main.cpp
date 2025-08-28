#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/losses.hpp"
#include "network/include/network.hpp"
#include "network/include/convolutional.hpp"
#include "dataset/include/mnist.hpp"
#include "network/include/reshape.hpp"
#include "network/include/sequential.hpp"
#include "network/include/sequential.hpp"
#include <iostream>
#include <vector>

int main(){
    torch::Tensor images = read_mnist_images("./data/t10k-images.idx3-ubyte");
    std::cout << images.sizes() << std::endl;
    torch::Tensor labels = read_mnist_labels("./data/t10k-labels.idx1-ubyte");
    std::cout << labels.sizes() << std::endl;
    
    torch::Tensor x = torch::reshape(images, {images.size(0), 1, 28, 28});
    torch::Tensor y = torch::one_hot(labels, 10).to(torch::kFloat64);
    y = torch::reshape(y, {y.size(0), 10, 1});
    
    Sequential sequential = Sequential::builder()
        .add<Convolutional>(std::vector<int64_t>{1, 28, 28}, 3, 5)
        .add<Sigmoid>()
        .add<Reshape>(std::vector<int64_t>{5, 26, 26}, std::vector<int64_t>{5*26*26, 1})
        .add<Dense>(5*26*26, 100)
        .add<Sigmoid>()
        .add<Dense>(100, 10)
        .add<Sigmoid>().build();
    

    Network network = Network(std::move(sequential), cross_entropy, cross_entropy_prime, 0.1, 50); 
    network.train(x, y, true);
    
    // for (int i = 0; i < x_test.size(0); i++){
    //     std::cout << network.eval(x_test[i]) << std::endl;
    // }
    
    return 0;
}
