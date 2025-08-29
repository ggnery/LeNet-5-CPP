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
#include "network/include/softmax.hpp"
#include <iostream>
#include <vector>

int main(){
    torch::Device device(
        #ifdef TORCH_CUDA_AVAILABLE
            torch::Device("cuda")
        #else
            torch::Device("cpu")
        #endif
    );
    std::cout << "device: " << device << std::endl;

    torch::Tensor images_test = read_mnist_images("./data/t10k-images.idx3-ubyte", device);
    torch::Tensor labels_test = read_mnist_labels("./data/t10k-labels.idx1-ubyte", device);
    torch::Tensor images_train = read_mnist_images("./data/train-images.idx3-ubyte", device);
    torch::Tensor labels_train = read_mnist_labels("./data/train-labels.idx1-ubyte", device);

    auto [x_train, x_test,  y_train, y_test] = preprocess_data(images_train, images_test, labels_train, labels_test);
    std::cout << "Train images sizes: "<< x_train.sizes() << std::endl;
    std::cout << "Train labels sizes: "<< y_train.sizes() << std::endl;
    std::cout << "Test images sizes: "<< x_test.sizes() << std::endl;
    std::cout << "Test labels sizes: "<< y_test.sizes() << std::endl;

    Sequential sequential = Sequential::builder()
        .add<Convolutional>(std::vector<int64_t>{1, 32, 32}, 5, 6)
        .add<Tanh>()
        .add<MaxPooling>(std::vector<int64_t>{2, 2}, 2)
        .add<Tanh>()
        .add<Convolutional>(std::vector<int64_t>{6, 14, 14}, 5, 16)
        .add<Tanh>()
        .add<MaxPooling>(std::vector<int64_t>{2, 2}, 2)
        .add<Tanh>()
        .add<Convolutional>(std::vector<int64_t>{16, 5, 5}, 5, 120)
        .add<Tanh>()
        .add<Reshape>(std::vector<int64_t>{120, 1, 1}, std::vector<int64_t>{120, 1})
        .add<Dense>(120, 84)
        .add<Tanh>()
        .add<Dense>(84, 10)
        .add<Softmax>()
        .build();
    
    Network network = Network(std::move(sequential), cross_entropy, cross_entropy_softmax_prime, 0.1, 2); 
    network.train(x_train, y_train, true);
    
    double accuracy = network.accuracy(x_test, y_test);
    std::cout << "Final model accuracy: " << accuracy << std::endl;
    return 0;
}
