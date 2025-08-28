#include "include/mnist.hpp"
#include <ATen/ops/pad.h>
#include <c10/core/TensorOptions.h>
#include <cstdlib>
#include <fstream>
#include <tuple>

// Convert big-endian to little-endian
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

torch::Tensor read_mnist_images(const std::string& images_path, torch::Device device){
    std::ifstream file(images_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "images file not found" << std::endl;
        std::exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t number_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_images, sizeof(number_images));
    number_images = reverse_int(number_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    torch::Tensor images = torch::empty({number_images, n_rows, n_cols}, torch::TensorOptions().dtype(torch::kFloat64).device(device));
    for(int i = 0; i<number_images; i++){
        for(int j = 0; j < n_rows; j++){
            for(int k = 0; k<n_cols; k++){
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                images[i][j][k] = temp/255.0; 
            }
        }
    }

    file.close();
    return images;
}

torch::Tensor read_mnist_labels(const std::string& labels_path, torch::Device device) {
    std::ifstream file(labels_path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "labels file not found" << std::endl;
        std::exit(1);
    }
    uint32_t magic_number = 0;
    uint32_t number_items = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_items, sizeof(number_items));
    number_items = reverse_int(number_items);

    torch::Tensor labels = torch::empty({number_items}, torch::TensorOptions().dtype(torch::kLong).device(device));
    for (int i = 0; i < number_items; i++){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = (long)temp;
    }

    file.close();
    return labels;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> preprocess_data(
    torch::Tensor train_images, 
    torch::Tensor test_images, 
    torch::Tensor train_labels, 
    torch::Tensor test_labels) {

        torch::Tensor x_train = torch::reshape(train_images, {train_images.size(0), 1, 28, 28});
        x_train = torch::pad(x_train, {2,2,2,2}); // transform (n_train_images,1,28,28) to  (n_images,1,32,32) by 0 padding

        torch::Tensor x_test = torch::reshape(test_images, {test_images.size(0), 1, 28, 28});
        x_test = torch::pad(x_test, {2,2,2,2});  // transform (n_test_images,1,28,28) to  (n_images,1,32,32) by 0 padding

        torch::Tensor y_train = torch::one_hot(train_labels, 10).to(torch::kFloat64);
        y_train = torch::reshape(y_train, {y_train.size(0), 10, 1});

        torch::Tensor y_test = torch::one_hot(test_labels, 10).to(torch::kFloat64);
        y_test = torch::reshape(y_test, {y_test.size(0), 10, 1});

        return {x_train, x_test, y_train, y_test};
}