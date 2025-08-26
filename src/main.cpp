#include "network/dense.hpp"
#include <iostream>

int main(){
    Dense<double> layer = Dense<double>(3, 3);
    std::cout << layer.bias;

    return 0;
}
