#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "activation.hpp"
#include "math_utils.hpp"

template<typename T>
class Tanh: public Activation<T>{
    public:

    //Constructor
    Tanh<T>(): Activation<T>(
        [](dlib::matrix<T> x) { return dlib::tanh(x); },
        [](dlib::matrix<T> x) { return tanh_prime(x); }
    ) {}

};

template<typename T>
class Sigmoid: public Activation<T>{
    public:

    // Constructor
    Sigmoid<T>(): Activation<T> (
        [](dlib::matrix<T> x) { return dlib::sigmoid(x); }, 
        [](dlib::matrix<T> x) { return sigmoid_prime(x); }
    ) {}
};

#endif // ACTIVATIONS_HPP