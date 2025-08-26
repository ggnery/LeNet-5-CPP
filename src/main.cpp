#include "network/dense.hpp"
#include "network/activations.hpp"
#include <iostream>

int main(){
    Dense<double> layer = Dense<double>(3, 3);
    Tanh<double> tanh = Tanh<double>();
    
    std::cout << layer.bias << std::endl << tanh.f_prime(layer.bias);

    return 0;
}
