#ifndef MNIST_HPP
#define MNIST_HPP

#include <cstdint>
#include <string>
#include <torch/torch.h>

uint32_t reverse_int(uint32_t i);
torch::Tensor read_mnist_images(const std::string& images_path);
torch::Tensor read_mnist_labels(const std::string& labels_path);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> preprocess_data(torch::Tensor train_images,  torch::Tensor test_images, torch::Tensor train_labels, torch::Tensor test_labels);

#endif // MNIST_HPP