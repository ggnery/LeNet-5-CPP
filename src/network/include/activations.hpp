#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "activation.hpp"

template<typename T>
class Tanh: public Activation<T>{
    public:

    Tanh<T>();
};

template<typename T>
class Sigmoid: public Activation<T>{
    public:

    Sigmoid<T>();
};

#endif // ACTIVATIONS_HPP