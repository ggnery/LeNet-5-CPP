#include "include/mnist.hpp"
#include <ATen/ops/empty.h>
#include <cstdlib>
#include <fstream>
#include <torch/types.h>

// Convert big-endian to little-endian
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

torch::Tensor read_mnist_images(const std::string& images_path){
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

    torch::Tensor images = torch::empty({number_images, n_rows, n_cols}, torch::kFloat64);
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

torch::Tensor read_mnist_labels(const std::string& labels_path) {
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

    torch::Tensor labels = torch::empty({number_items}, torch::kLong);
    for (int i = 0; i < number_items; i++){
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = (long)temp;
    }

    file.close();
    return labels;
}