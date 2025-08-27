#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/losses.hpp"
#include <iostream>

int main(){
    Dense layer = Dense(3, 3);
    Tanh tanh = Tanh();
    auto a = layer.bias;
    auto b = tanh.f_prime(layer.bias);

    std::cout << cross_entropy_prime(a, b);

    return 0;
}
