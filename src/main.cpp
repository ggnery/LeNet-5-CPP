#include "network/include/dense.hpp"
#include "network/include/activations.hpp"
#include "network/include/losses.hpp"
#include <iostream>

int main(){
    Dense<long double> layer = Dense<long double>(3, 3);
    Tanh<long double> tanh = Tanh<long double>();
    auto a = layer.bias;
    auto b = tanh.f_prime(layer.bias);

    std::cout << cross_entropy(a, b);

    return 0;
}
