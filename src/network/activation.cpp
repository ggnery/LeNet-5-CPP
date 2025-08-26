#include "include/activation.hpp"

template class Activation<double>;
template class Activation<float>;
template class Activation<long double>;

//Constructor
template<typename T>
Activation<T>::Activation(std::function<dlib::matrix<T>(dlib::matrix<T>)> f, std::function<dlib::matrix<T>(dlib::matrix<T>)> f_prime) {
    this->f = f;
    this->f_prime = f_prime;
}

template<typename T>
dlib::matrix<T> Activation<T>::forward(dlib::matrix<T> input) {
    this->input = input;
    return this->f(input); // Y = f(X) 
}

template<typename T>
dlib::matrix<T> Activation<T>::backward(dlib::matrix<T> output_gradient, double eta) {
    return dlib::pointwise_multiply(output_gradient, this->f_prime(this->input)); // ∂E/∂X = ∂E/∂Y ⊙ f'(X)
}