#include "activations.hpp"
#include "math_utils.hpp"
#include "activation.hpp"


template class Tanh<float>;
template class Tanh<double>;
template class Tanh<long double>;

template class Sigmoid<float>;
template class Sigmoid<double>;
template class Sigmoid<long double>;

template <typename T>
Tanh<T>::Tanh(): Activation<T>(
    [](dlib::matrix<T> x) { return dlib::tanh(x); },
    [](dlib::matrix<T> x) { return tanh_prime(x); }
) {}

template <typename T>
Sigmoid<T>::Sigmoid(): Activation<T> (
    [](dlib::matrix<T> x) { return dlib::sigmoid(x); }, 
    [](dlib::matrix<T> x) { return sigmoid_prime(x); }
) {}